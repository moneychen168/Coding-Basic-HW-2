// Planner (agent-P). Host-only; decides per problem: emulate vs native, num_moduli, tile_cfg.
#include "planner.h"
#include "internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double env_or(const char* name, double dflt) {
  const char* s = std::getenv(name);
  if (!s || !*s) return dflt;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  return (end != s && v > 0) ? v : dflt;
}

// Minimal reader for the scalar fields of bench/calib params JSON ("key": number).
static bool json_num(const char* txt, const char* key, double* v) {
  char pat[96]; std::snprintf(pat, sizeof pat, "\"%s\"", key);
  const char* p = std::strstr(txt, pat);
  if (!p) return false;
  p = std::strchr(p + std::strlen(pat), ':');
  if (!p) return false;
  char* end = nullptr; double x = std::strtod(p + 1, &end);
  if (end == p + 1 || !(x > 0)) return false;
  *v = x; return true;
}

// params file: env GGEMM_CALIB_JSON, else <GGEMM_CALIB_DIR>/params_<slug>.json where slug
// follows bench/calib/fit.py: non-alnum runs -> '_', trimmed, lowercased ("NVIDIA H200" ->
// "nvidia_h200"). Missing file -> nominal defaults stay.
static void load_calib_json(const char* gpu_name, double* fp64, double* i8, double* bw) {
  char path[1024];
  const char* env = std::getenv("GGEMM_CALIB_JSON");
  if (env && *env) {
    std::snprintf(path, sizeof path, "%s", env);
  } else {
#ifdef GGEMM_CALIB_DIR
    char slug[256]; size_t n = 0; bool sep = false;
    for (const char* c = gpu_name ? gpu_name : ""; *c && n + 1 < sizeof slug; ++c) {
      if (std::isalnum((unsigned char)*c)) {
        if (sep && n) slug[n++] = '_';
        sep = false; slug[n++] = (char)std::tolower((unsigned char)*c);
      } else sep = true;
    }
    slug[n] = 0;
    std::snprintf(path, sizeof path, "%s/params_%s.json", GGEMM_CALIB_DIR, slug);
#else
    return;
#endif
  }
  FILE* f = std::fopen(path, "rb");
  if (!f) return;
  static char buf[1 << 16];           // scalars sit at the top of the file
  size_t n = std::fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; std::fclose(f);
  json_num(buf, "f64_peak_flops", fp64);
  json_num(buf, "i8_peak_ops", i8);
  json_num(buf, "bw_Bps", bw);
}

// Fallback defaults = nominal datasheet peak x an efficiency factor (NOT measured).
void ggemm_device_model_init(ggemm_device_model_t* dm, const char* gpu_name, int cc_major,
                             int cc_minor, int num_sms) {
  (void)cc_minor;
  double fp64, i8, bw;
  if (cc_major == 9) {          // H100/H200 SXM nominal: FP64 TC 67 TF, INT8 1979 TOPS dense, 3.35 TB/s
    fp64 = 67e12 * 0.8;  i8 = 1979e12 * 0.5;  bw = 3.35e12 * 0.8;
  } else if (cc_major == 12) {  // RTX 5090 nominal: FP64 = FP32/64 ~1.64 TF; INT8 dense (unverified) ~838 TOPS; 1.79 TB/s
    fp64 = 1.64e12 * 0.8; i8 = 838e12 * 0.5;  bw = 1.79e12 * 0.8;
  } else {                      // unknown arch: conservative guess, calibrate via env
    fp64 = 10e12;  i8 = 200e12;  bw = 1e12;
  }
  load_calib_json(gpu_name, &fp64, &i8, &bw);
  dm->num_sms = num_sms > 0 ? num_sms : 1;
  dm->fp64_flops = env_or("GGEMM_FP64_FLOPS", fp64);
  dm->int8_ops   = env_or("GGEMM_INT8_OPS", i8);
  dm->mem_bw     = env_or("GGEMM_MEM_BW", bw);
  dm->tiny_mnk   = env_or("GGEMM_TINY_MNK", 32.0 * 32.0 * 32.0);
}

// Accuracy target definition used here: target_rel_err is a NORMWISE bound
//   |C_ij - (AB)_ij| <= target * ||A(i,:)||_2 ||B(:,j)||_2   (default 1e-15),
// i.e. the quantity oz2_error_bound(k, N) bounds. The preferred per-problem target
// K*u/rho_hat (rho_hat = span proxy measured by prep) is NOT used: the span is only known
// on the device after prep, and feeding it back to the host planner would need a D2H +
// stream sync inside every timed call (and break the 3-launch pipeline). Limitation:
// N is chosen from (k, target) alone, so it may over-provision for low-span inputs.
int ggemm_plan_num_moduli(const ggemm_config_t* cfg, int k) {
  int nm;
  if (cfg && cfg->num_moduli > 0) {
    nm = cfg->num_moduli;
  } else {
    double tgt = (cfg && cfg->target_rel_err > 0) ? cfg->target_rel_err : 1e-15;
    nm = oz2_choose_num_moduli(k, tgt);
    if (nm < 0) return -1;  // target unreachable: NEVER clamp to 2 (~8-bit); caller falls back / errors
  }
  return std::min(std::max(nm, 2), GGEMM_MAX_MODULI);
}

static inline size_t kp_of(int k) { return ggemm_align_up((size_t)k, GGEMM_KP_ALIGN); }
static inline double rup(double x, double t) { return std::ceil(x / t) * t; }

// Largest tile whose padded-area efficiency m*n/(mp*np) >= 0.75; else the most efficient one.
static int tile_by_padding(int m, int n) {
  int best = GGEMM_NUM_TILE_CFG - 1; double best_eff = -1;
  for (int c = 0; c < GGEMM_NUM_TILE_CFG; ++c) {
    double t = ggemm_tile_dim(c);
    double eff = (double)m * n / (rup(m, t) * rup(n, t));
    if (eff >= 0.75) return c;
    if (eff > best_eff) { best_eff = eff; best = c; }
  }
  return best;
}

static double tiles_of(int m, int n, int cfg) {
  double t = ggemm_tile_dim(cfg);
  return std::ceil(m / t) * std::ceil(n / t);
}

static void cost(const ggemm_problem_t& p, int nm, int cfg, const ggemm_device_model_t* dm,
                 double* t_emu, double* t_nat) {
  double m = p.m, n = p.n, k = p.k, kp = (double)kp_of(p.k), t = ggemm_tile_dim(cfg);
  double c_rw = (p.beta != 0.0) ? 2.0 : 1.0;
  // emulation = three dependent stages; byte terms match bench/calib/fit.py (incl. the
  // INT32 C32 write of the GEMM). Launch latency is omitted: in a grouped call the 3
  // emulation launches / 1 native launch are shared by the whole group, not per problem.
  // INT8 work / residue + C32 traffic scale with the plane count N + OZ2_EXTRA_PLANES (abs plane).
  const double np = oz2_num_planes(nm);
  double prep  = ((m * k + k * n) * 8 + np * (m + n) * kp) / dm->mem_bw;
  double gemm  = np * 2.0 * rup(m, t) * rup(n, t) * kp / dm->int8_ops + np * m * n * 4 / dm->mem_bw;
  double recon = (np * m * n * 4 + c_rw * m * n * 8) / dm->mem_bw;
  *t_emu = prep + gemm + recon;
  double nat_c = 2.0 * m * n * k / dm->fp64_flops;
  double nat_m = (m * k + k * n + c_rw * m * n) * 8 / dm->mem_bw;
  *t_nat = std::max(nat_c, nat_m);
}

// Wave quantization: a persistent grid holds ~num_sms tiles at once (assumes 1 CTA/SM;
// adjust if agent-K reports higher occupancy). Makespan >= max(W/slots, largest tile).
// If one problem's tile is heavier than the per-slot average, its tiles are too few to
// fill the machine -> shrink its tile (more, smaller tiles). Repeat until stable.
static void tune_tiles(const ggemm_problem_t* probs, int num, const ggemm_device_model_t* dm,
                       ggemm_plan_entry_t* out) {
  const double slots = dm->num_sms;
  for (int pass = 0; pass < GGEMM_NUM_TILE_CFG; ++pass) {
    double W = 0;
    for (int g = 0; g < num; ++g) {
      if (out[g].route != GGEMM_ROUTE_EMU) continue;
      double t = ggemm_tile_dim(out[g].tile_cfg);
      W += oz2_num_planes(out[g].num_moduli) * tiles_of(probs[g].m, probs[g].n, out[g].tile_cfg) * t * t * kp_of(probs[g].k);
    }
    if (W <= 0) return;
    const double per_slot = W / slots;
    bool changed = false;
    for (int g = 0; g < num; ++g) {
      if (out[g].route != GGEMM_ROUTE_EMU || out[g].tile_cfg >= GGEMM_NUM_TILE_CFG - 1) continue;
      double t = ggemm_tile_dim(out[g].tile_cfg);
      if (t * t * kp_of(probs[g].k) > per_slot) { out[g].tile_cfg++; changed = true; }
    }
    if (!changed) return;
  }
}

int ggemm_plan(const ggemm_problem_t* probs, int num, const ggemm_config_t* cfg,
               ggemm_path_t path, const ggemm_device_model_t* dm, ggemm_plan_entry_t* out) {
  const int allow_fb = cfg ? cfg->allow_fallback : 0;
  const bool autop = (path == GGEMM_PATH_AUTO);
  static const double rho_budget = env_or("GGEMM_RHO_BUDGET", GGEMM_RHO_BUDGET_DEFAULT);
  for (int g = 0; g < num; ++g) {
    const ggemm_problem_t& p = probs[g];
    ggemm_plan_entry_t& e = out[g];
    e.route = GGEMM_ROUTE_EMU; e.num_moduli = 0; e.tile_cfg = 0; e.t_emu_s = e.t_native_s = 0;
    if (p.m == 0 || p.n == 0) { e.route = GGEMM_ROUTE_EMPTY; continue; }
    if (p.k == 0) { e.route = GGEMM_ROUTE_K_ZERO; continue; }
    if (p.k > GGEMM_MAX_K_EXACT) {
      if (!allow_fb) return GGEMM_ERR_UNSAFE;
      e.route = GGEMM_ROUTE_K_TOO_LARGE; continue;
    }
    if (autop && allow_fb && p.k <= GGEMM_AUTO_NATIVE_MAX_K) { e.route = GGEMM_ROUTE_SMALL_K; continue; }
    if (autop && !(cfg && cfg->num_moduli > 0)) {
      // AUTO adaptive: componentwise planning (uncertified entries are repaired in FP64, so N
      // only moves the repair rate). -1 only for k > OZ2_MAX_K (already rerouted above).
      e.num_moduli = oz2_choose_num_moduli_cw(p.k, rho_budget);
      if (e.num_moduli >= 0) e.num_moduli = std::min(std::max(e.num_moduli, 2), GGEMM_MAX_MODULI);
    } else {
      e.num_moduli = ggemm_plan_num_moduli(cfg, p.k);
    }
    if (e.num_moduli < 0) {
      if (!allow_fb) return GGEMM_ERR_UNSAFE;
      e.route = GGEMM_ROUTE_PRECISION; e.num_moduli = 0; continue;
    }
    if (!autop) continue;  // EMU_LOOP / EMU_GROUPED: fixed tile 0, no cost-based fallback
    e.tile_cfg = tile_by_padding(p.m, p.n);
    cost(p, e.num_moduli, e.tile_cfg, dm, &e.t_emu_s, &e.t_native_s);
    if (allow_fb) {
      if ((double)p.m * p.n * p.k < dm->tiny_mnk) e.route = GGEMM_ROUTE_TINY;
      else if (e.t_emu_s > e.t_native_s) e.route = GGEMM_ROUTE_UNPROFITABLE;
    }
  }
  if (autop) tune_tiles(probs, num, dm, out);
  return GGEMM_OK;
}
