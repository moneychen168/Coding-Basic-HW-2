// Planner (agent-P). Host-only; decides per problem: emulate vs native, num_moduli, tile_cfg.
#include "planner.h"
#include "internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

static double env_or(const char* name, double dflt) {
  const char* s = std::getenv(name);
  if (!s || !*s) return dflt;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  return (end != s && v > 0) ? v : dflt;
}

// Plausible ranges of the throughput model; values outside are rejected (warning on stderr).
struct calib_range { const char* key; double lo, hi; };
static const calib_range kRangeFP64 = {"fp64_flops", 1e11, 1e16};
static const calib_range kRangeI8   = {"int8_ops",   1e12, 1e17};
static const calib_range kRangeBW   = {"mem_bw",     1e10, 1e14};

static bool in_range(const calib_range& r, double v, const char* src) {
  if (v >= r.lo && v <= r.hi) return true;
  std::fprintf(stderr, "[ggemm] WARNING: %s %s=%g outside [%g, %g], ignored\n", src, r.key, v,
               r.lo, r.hi);
  return false;
}

// env override with range check (keeps *v if unset / unparsable / out of range); true if used.
static bool env_range(const char* name, const calib_range& r, double* v) {
  const char* s = std::getenv(name);
  if (!s || !*s) return false;
  char* end = nullptr;
  double x = std::strtod(s, &end);
  if (end == s || !in_range(r, x, name)) return false;
  *v = x;
  return true;
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
// used: receives the path actually read (left untouched when no file was read);
// got[0..2]: fp64 / i8 / bw taken from the file (present and in range).
static void load_calib_json(const char* gpu_name, double* fp64, double* i8, double* bw,
                            char* used, size_t used_sz, bool* got) {
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
  double v;
  if (json_num(buf, "f64_peak_flops", &v) && in_range(kRangeFP64, v, path)) { *fp64 = v; got[0] = true; }
  if (json_num(buf, "i8_peak_ops", &v) && in_range(kRangeI8, v, path)) { *i8 = v; got[1] = true; }
  if (json_num(buf, "bw_Bps", &v) && in_range(kRangeBW, v, path)) { *bw = v; got[2] = true; }
  if (used) std::snprintf(used, used_sz, "%s", path);
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
  char used[1024] = "none";
  bool got[3] = {false, false, false};
  load_calib_json(gpu_name, &fp64, &i8, &bw, used, sizeof used, got);
  const char* src[4];
  for (int i = 0; i < 3; ++i) src[i] = got[i] ? "json" : "nominal";
  if (env_range("GGEMM_FP64_FLOPS", kRangeFP64, &fp64)) src[0] = "env";
  if (env_range("GGEMM_INT8_OPS", kRangeI8, &i8)) src[1] = "env";
  if (env_range("GGEMM_MEM_BW", kRangeBW, &bw)) src[2] = "env";
  dm->num_sms = num_sms > 0 ? num_sms : 1;
  dm->fp64_flops = fp64; dm->int8_ops = i8; dm->mem_bw = bw;
  // GGEMM_TINY_MNK=0 disables the tiny route (env_or would reject 0).
  dm->tiny_mnk = 32.0 * 32.0 * 32.0; src[3] = "default";
  if (const char* s = std::getenv("GGEMM_TINY_MNK")) {
    char* end = nullptr; double v = std::strtod(s, &end);
    if (end != s && v >= 0) { dm->tiny_mnk = v; src[3] = "env"; }
  }
  static bool printed = false;          // one line per process, after env + range checks
  if (!printed) {
    printed = true;
    std::fprintf(stderr, "[ggemm] calib fp64_flops=%.3g(%s) int8_ops=%.3g(%s) mem_bw=%.3g(%s) "
                 "tiny_mnk=%.3g(%s) json=%s\n", fp64, src[0], i8, src[1], bw, src[2],
                 dm->tiny_mnk, src[3], used);
  }
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

#define GGEMM_PREP_PASSES 2.0
static const double kTileEff[GGEMM_NUM_TILE_CFG] = {1.0, 0.63, 0.30};   // code review (2)
// Resident CTAs per SM of agent-K's persistent kernel. Constant until agent-K exposes its
// occupancy query (internal dev_info().ctas_per_sm); review L1 asks to switch to that.
#define GGEMM_I8_CTAS_PER_SM 2

static void cost(const ggemm_problem_t& p, int nm, int cfg, const ggemm_device_model_t* dm,
                 double* t_emu, double* t_nat) {
  double m = p.m, n = p.n, k = p.k, kp = (double)kp_of(p.k), t = ggemm_tile_dim(cfg);
  double c_rw = (p.beta != 0.0) ? 2.0 : 1.0;
  // emulation = three dependent stages; byte terms match bench/calib/fit.py (incl. the
  // INT32 C32 write of the GEMM). Launch latency is omitted: in a grouped call the 3
  // emulation launches / 1 native launch are shared by the whole group, not per problem.
  // INT8 work / residue + C32 traffic scale with the plane count N + OZ2_EXTRA_PLANES (abs plane).
  const double np = oz2_num_planes(nm);
  // prep reads FP64 A and B GGEMM_PREP_PASSES times (norm pass + residue pass; a third pass only
  // for rows/columns needing rescale, which the host cannot know -> not modelled), then writes
  // np residue planes + OZ2_ROW_WS_PER_ROW int32 per row/column.
  double prep  = (GGEMM_PREP_PASSES * (m * k + k * n) * 8 + np * (m + n) * kp +
                  4.0 * OZ2_ROW_WS_PER_ROW * (m + n)) / dm->mem_bw;
  // effective INT8 rate per tile config (smaller tiles: lower MMA efficiency)
  // Wave quantization (review M3): the persistent grid runs `slots` tiles at a time, each slot
  // at int8_ops * eff / slots; np * tiles tiles take ceil(np * tiles / slots) waves.
  const double slots = (double)dm->num_sms * GGEMM_I8_CTAS_PER_SM;
  const double per_tile = 2.0 * t * t * kp / (dm->int8_ops * kTileEff[cfg] / slots);
  double gemm  = std::ceil(np * tiles_of(p.m, p.n, cfg) / slots) * per_tile +
                 np * m * n * 4 / dm->mem_bw;
  double recon = (np * m * n * 4 + c_rw * m * n * 8) / dm->mem_bw;
  *t_emu = prep + gemm + recon;
  double nat_c = 2.0 * m * n * k / dm->fp64_flops;
  double nat_m = (m * k + k * n + c_rw * m * n) * 8 / dm->mem_bw;
  *t_nat = std::max(nat_c, nat_m);
}

// Wave quantization: a persistent grid holds ~num_sms * GGEMM_I8_CTAS_PER_SM tiles at once. Makespan >= max(W/slots, largest tile).
// If one problem's tile is heavier than the per-slot average, its tiles are too few to
// fill the machine -> shrink its tile (more, smaller tiles). Repeat until stable.
// Returns true if any tile_cfg was changed (the caller re-costs only then).
static bool tune_tiles(const ggemm_problem_t* probs, int num, const ggemm_device_model_t* dm,
                       ggemm_plan_entry_t* out) {
  const double slots = (double)dm->num_sms * GGEMM_I8_CTAS_PER_SM;
  bool any = false;
  for (int pass = 0; pass < GGEMM_NUM_TILE_CFG; ++pass) {
    double W = 0;
    for (int g = 0; g < num; ++g) {
      if (out[g].route != GGEMM_ROUTE_EMU) continue;
      double t = ggemm_tile_dim(out[g].tile_cfg);
      W += oz2_num_planes(out[g].num_moduli) * tiles_of(probs[g].m, probs[g].n, out[g].tile_cfg) * t * t * kp_of(probs[g].k);
    }
    if (W <= 0) return any;
    const double per_slot = W / slots;
    bool changed = false;
    for (int g = 0; g < num; ++g) {
      if (out[g].route != GGEMM_ROUTE_EMU || out[g].tile_cfg >= GGEMM_NUM_TILE_CFG - 1) continue;
      double t = ggemm_tile_dim(out[g].tile_cfg);
      if (t * t * kp_of(probs[g].k) > per_slot) { out[g].tile_cfg++; changed = true; }
    }
    if (!changed) return any;
    any = true;
  }
  return any;
}

// Everything below the alpha == 0 check depends on the problem only through
// (m, n, k, beta != 0) -- cfg, path, dm and rho_budget are per call. plan_shape() is that part
// (e already reset to route EMU by the caller); the memo in ggemm_plan_ex keys on exactly these.
static int plan_shape(const ggemm_problem_t& p, const ggemm_config_t* cfg, bool autop, int allow_fb,
                      double rho_budget, const ggemm_device_model_t* dm, ggemm_plan_entry_t& e) {
  if (p.k > GGEMM_MAX_K_EXACT) {
    if (!allow_fb) return GGEMM_ERR_UNSAFE;
    e.route = GGEMM_ROUTE_K_TOO_LARGE; return GGEMM_OK;
  }
  if (autop && allow_fb && p.k <= GGEMM_AUTO_NATIVE_MAX_K) { e.route = GGEMM_ROUTE_SMALL_K; return GGEMM_OK; }
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
    e.route = GGEMM_ROUTE_PRECISION; e.num_moduli = 0; return GGEMM_OK;
  }
  if (!autop) return GGEMM_OK;  // EMU_LOOP / EMU_GROUPED: fixed tile 0, no cost-based fallback
  e.tile_cfg = tile_by_padding(p.m, p.n);
  cost(p, e.num_moduli, e.tile_cfg, dm, &e.t_emu_s, &e.t_native_s);
  if (allow_fb) {
    if ((double)p.m * p.n * p.k < dm->tiny_mnk) e.route = GGEMM_ROUTE_TINY;
    else if (e.t_emu_s > e.t_native_s) e.route = GGEMM_ROUTE_UNPROFITABLE;
  }
  return GGEMM_OK;
}

static inline bool same_shape(const ggemm_problem_t& a, const ggemm_problem_t& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k && (a.beta != 0.0) == (b.beta != 0.0);
}
static inline uint32_t shape_hash(const ggemm_problem_t& p) {
  uint32_t h = (uint32_t)p.m * 73856093u ^ (uint32_t)p.n * 19349663u ^ (uint32_t)p.k * 83492791u ^
               (p.beta != 0.0 ? 2654435761u : 0u);
  h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
  return h;
}

size_t ggemm_plan_memo_cap(int max_problems) {
  size_t cap = 1;
  while (cap < 2 * (size_t)std::max(max_problems, 1)) cap <<= 1;
  return cap;
}

int ggemm_plan(const ggemm_problem_t* probs, int num, const ggemm_config_t* cfg,
               ggemm_path_t path, const ggemm_device_model_t* dm, ggemm_plan_entry_t* out) {
  return ggemm_plan_ex(probs, num, cfg, path, dm, out, nullptr, 0);
}

int ggemm_plan_ex(const ggemm_problem_t* probs, int num, const ggemm_config_t* cfg,
                  ggemm_path_t path, const ggemm_device_model_t* dm, ggemm_plan_entry_t* out,
                  int* memo, size_t memo_cap) {
  const int allow_fb = cfg ? cfg->allow_fallback : 0;
  const bool autop = (path == GGEMM_PATH_AUTO);
  static const double rho_budget = env_or("GGEMM_RHO_BUDGET", GGEMM_RHO_BUDGET_DEFAULT);
  // Shape memo: open-addressing table of problem indices (-1 = empty), linear probing; the
  // first problem of each distinct shape is planned, later ones copy its entry bit for bit.
  // Caller-owned (ggemm_create sizes it), so no allocation here; too small -> no memo.
  size_t mask = 0;
  if (memo && num > 0) {
    const size_t cap = ggemm_plan_memo_cap(num);
    if (cap <= memo_cap) { std::fill(memo, memo + cap, -1); mask = cap - 1; }
    else memo = nullptr;
  }
  for (int g = 0; g < num; ++g) {
    const ggemm_problem_t& p = probs[g];
    ggemm_plan_entry_t& e = out[g];
    e.route = GGEMM_ROUTE_EMU; e.num_moduli = 0; e.tile_cfg = 0; e.t_emu_s = e.t_native_s = 0;
    if (p.m == 0 || p.n == 0) { e.route = GGEMM_ROUTE_EMPTY; continue; }
    if (p.k == 0) { e.route = GGEMM_ROUTE_K_ZERO; continue; }
    // alpha == 0: C = beta*C exactly; never emulate (prep would leave planes unwritten).
    if (p.alpha == 0.0) { e.route = GGEMM_ROUTE_ALPHA_ZERO; continue; }
    if (memo) {
      size_t s = shape_hash(p) & mask;
      int rep;
      while ((rep = memo[s]) >= 0 && !same_shape(probs[rep], p)) s = (s + 1) & mask;
      if (rep >= 0) { e = out[rep]; continue; }   // rep < g: its entry is complete
      memo[s] = g;
    }
    const int r = plan_shape(p, cfg, autop, allow_fb, rho_budget, dm, e);
    if (r) return r;
  }
  // tile changes move t_emu: re-evaluate cost and the profitability route once. Skipped when
  // tune_tiles changed nothing: cost() would then reproduce the first-pass values exactly.
  if (autop && tune_tiles(probs, num, dm, out)) {
    for (int g = 0; g < num; ++g) {
      ggemm_plan_entry_t& e = out[g];
      if (e.route != GGEMM_ROUTE_EMU) continue;
      cost(probs[g], e.num_moduli, e.tile_cfg, dm, &e.t_emu_s, &e.t_native_s);
      if (allow_fb && e.t_emu_s > e.t_native_s) e.route = GGEMM_ROUTE_UNPROFITABLE;
    }
  }
  return GGEMM_OK;
}
