// tests/test_planner.cpp -- host-only, table-driven unit tests for src/planner.cpp (AUTO planner).
// Not part of the CMake test glob (that only takes tests/test_*.cu); built and run by
// scripts/run_planner_unit.slurm. Pure host code: no kernel launches, no device memory.
//
// Usage: test_planner [testdata_dir]   (default: tests/testdata/planner, relative to repo root)
// Exit code: 0 = all checks passed, 1 = at least one FAIL.
//
// Interface assumed (src/planner.h, src/internal.h, include/ggemm.h as of 2026-10-07):
//   ggemm_device_model_init(dm, gpu_name, cc_major, cc_minor, num_sms)
//   ggemm_plan(probs, num, cfg, path, dm, out) -> GGEMM_OK / GGEMM_ERR_UNSAFE
//   ggemm_plan_entry_t { route, num_moduli, tile_cfg, t_emu_s, t_native_s }
//   ggemm_device_model_t { num_sms, fp64_flops, int8_ops, mem_bw, tiny_mnk, ... }
//   routes GGEMM_ROUTE_{EMU,EMPTY,K_ZERO,K_TOO_LARGE,TINY,UNPROFITABLE,PRECISION,SMALL_K,ALPHA_ZERO}
//   ggemm_tile_dim(cfg), GGEMM_NUM_TILE_CFG, GGEMM_MAX_K_EXACT, GGEMM_AUTO_NATIVE_MAX_K,
//   GGEMM_RHO_BUDGET_DEFAULT, oz2_choose_num_moduli_cw, oz2_num_planes.
#include "planner.h"
#include "internal.h"
#include "ozaki2_numerics.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

// GCC 11 flags free() on memory from the replaced operator new as mismatched; it is not.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
// ---------------------------------------------------------------- allocation counting hook
static std::atomic<long> g_new_calls{0};
static std::atomic<bool> g_count_new{false};
void* operator new(std::size_t sz) {
  if (g_count_new.load(std::memory_order_relaxed)) g_new_calls.fetch_add(1);
  if (void* p = std::malloc(sz ? sz : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t sz) { return operator new(sz); }
void* operator new(std::size_t sz, const std::nothrow_t&) noexcept {
  if (g_count_new.load(std::memory_order_relaxed)) g_new_calls.fetch_add(1);
  return std::malloc(sz ? sz : 1);
}
void* operator new[](std::size_t sz, const std::nothrow_t& t) noexcept { return operator new(sz, t); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// ---------------------------------------------------------------- tiny check framework
static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    if (cond) { ++g_pass; }                                                 \
    else { ++g_fail; std::printf("  FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond); \
           std::printf(__VA_ARGS__); std::printf("\n"); }                   \
  } while (0)

static const char* route_name(int r) {
  switch (r) {
    case GGEMM_ROUTE_EMU: return "EMU";           case GGEMM_ROUTE_EMPTY: return "EMPTY";
    case GGEMM_ROUTE_K_ZERO: return "K_ZERO";     case GGEMM_ROUTE_K_TOO_LARGE: return "K_TOO_LARGE";
    case GGEMM_ROUTE_TINY: return "TINY";         case GGEMM_ROUTE_UNPROFITABLE: return "UNPROFITABLE";
    case GGEMM_ROUTE_PRECISION: return "PRECISION"; case GGEMM_ROUTE_SMALL_K: return "SMALL_K";
    case GGEMM_ROUTE_ALPHA_ZERO: return "ALPHA_ZERO";
    default: return "?";
  }
}

static ggemm_problem_t P(int m, int n, int k, double alpha = 1.0, double beta = 0.0) {
  ggemm_problem_t p;
  std::memset(&p, 0, sizeof p);
  p.m = m; p.n = n; p.k = k; p.lda = m > 0 ? m : 1; p.ldb = k > 0 ? k : 1; p.ldc = m > 0 ? m : 1;
  p.alpha = alpha; p.beta = beta;   // A/B/C stay nullptr: the planner must never dereference them
  return p;
}
static ggemm_config_t C(int nm, int fb, double tgt = 1e-15) {
  ggemm_config_t c;
  std::memset(&c, 0, sizeof c);
  c.num_moduli = nm; c.target_rel_err = tgt; c.allow_fallback = fb;
  return c;
}

static const char* kEnvVars[] = {"GGEMM_FP64_FLOPS", "GGEMM_INT8_OPS", "GGEMM_MEM_BW",
                                 "GGEMM_TINY_MNK",   "GGEMM_CALIB_JSON", "GGEMM_RHO_BUDGET"};
static void clean_env() { for (const char* v : kEnvVars) unsetenv(v); }

// H200-like model from a fixed calib file (deterministic, independent of bench/calib changes).
static std::string g_dir;
static ggemm_device_model_t model_h200(int num_sms = 132) {
  ggemm_device_model_t dm;
  std::memset(&dm, 0, sizeof dm);
  setenv("GGEMM_CALIB_JSON", (g_dir + "/valid.json").c_str(), 1);
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, num_sms);
  unsetenv("GGEMM_CALIB_JSON");
  return dm;
}

// Run fn with stderr redirected to a temp file; returns what was written to stderr.
template <class F> static std::string capture_stderr(F fn) {
  std::fflush(stderr);
  FILE* tmp = std::tmpfile();
  int saved = dup(2);
  dup2(fileno(tmp), 2);
  fn();
  std::fflush(stderr);
  dup2(saved, 2); close(saved);
  std::string s; char buf[512]; std::rewind(tmp);
  while (size_t n = std::fread(buf, 1, sizeof buf, tmp)) s.append(buf, n);
  std::fclose(tmp);
  return s;
}

static double pad_eff(int m, int n, int cfg) {
  double t = ggemm_tile_dim(cfg);
  return (double)m * n / (std::ceil(m / t) * t * std::ceil(n / t) * t);
}

// ================================================================= 1. routes (table-driven)
struct RouteCase {
  const char* name; ggemm_problem_t p; int nm, fb; ggemm_path_t path;
  int want_rc;      // GGEMM_OK / GGEMM_ERR_UNSAFE
  int want_route;   // checked only when want_rc == GGEMM_OK; -1 = "anything but SMALL_K"
};
static void test_routes(const ggemm_device_model_t& dm) {
  std::printf("[routes]\n");
  const ggemm_path_t AU = GGEMM_PATH_AUTO, EG = GGEMM_PATH_EMU_GROUPED, EL = GGEMM_PATH_EMU_LOOP;
  const int KMAX = GGEMM_MAX_K_EXACT, KS = GGEMM_AUTO_NATIVE_MAX_K;
  const RouteCase cases[] = {
    {"K>max, AUTO fb=1 -> K_TOO_LARGE",        P(256, 256, KMAX + 1), 0, 1, AU, GGEMM_OK, GGEMM_ROUTE_K_TOO_LARGE},
    {"K>max, EMU_GROUPED fb=1 -> K_TOO_LARGE", P(256, 256, KMAX + 1), 0, 1, EG, GGEMM_OK, GGEMM_ROUTE_K_TOO_LARGE},
    {"K>max fixed N, fb=1 -> K_TOO_LARGE",     P(256, 256, 200000),   8, 1, AU, GGEMM_OK, GGEMM_ROUTE_K_TOO_LARGE},
    {"K>max, AUTO fb=0 -> ERR_UNSAFE",         P(256, 256, KMAX + 1), 0, 0, AU, GGEMM_ERR_UNSAFE, 0},
    {"K>max, EMU_LOOP fb=0 -> ERR_UNSAFE",     P(256, 256, KMAX + 1), 0, 0, EL, GGEMM_ERR_UNSAFE, 0},
    {"K>max fixed N, fb=0 -> ERR_UNSAFE",      P(256, 256, KMAX + 1), 8, 0, EG, GGEMM_ERR_UNSAFE, 0},
    {"K==max, AUTO fb=0 -> EMU",               P(256, 256, KMAX),     0, 0, AU, GGEMM_OK, GGEMM_ROUTE_EMU},
    {"k=1  AUTO fb=1 -> SMALL_K",              P(512, 512, 1),        0, 1, AU, GGEMM_OK, GGEMM_ROUTE_SMALL_K},
    {"k=16 AUTO fb=1 -> SMALL_K",              P(512, 512, KS),       0, 1, AU, GGEMM_OK, GGEMM_ROUTE_SMALL_K},
    {"k=17 AUTO fb=1 -> not SMALL_K",          P(512, 512, KS + 1),   0, 1, AU, GGEMM_OK, -1},
    {"k=8  AUTO fb=0 -> EMU (no fallback)",    P(512, 512, 8),        0, 0, AU, GGEMM_OK, GGEMM_ROUTE_EMU},
    {"k=8  EMU_GROUPED fb=1 -> EMU",           P(512, 512, 8),        0, 1, EG, GGEMM_OK, GGEMM_ROUTE_EMU},
    {"alpha=0 AUTO fb=1 -> ALPHA_ZERO",        P(512, 512, 512, 0.0, 1.0), 0, 1, AU, GGEMM_OK, GGEMM_ROUTE_ALPHA_ZERO},
    {"alpha=0 AUTO fb=0 -> ALPHA_ZERO",        P(512, 512, 512, 0.0, 0.5), 0, 0, AU, GGEMM_OK, GGEMM_ROUTE_ALPHA_ZERO},
    {"alpha=0 EMU_GROUPED -> ALPHA_ZERO",      P(512, 512, 512, 0.0, 0.0), 4, 0, EG, GGEMM_OK, GGEMM_ROUTE_ALPHA_ZERO},
    {"alpha=0 k=8 AUTO fb=1 -> ALPHA_ZERO",    P(512, 512, 8, 0.0, 1.0),   0, 1, AU, GGEMM_OK, GGEMM_ROUTE_ALPHA_ZERO},
    {"m=0 -> EMPTY",                           P(0, 512, 512),        0, 0, AU, GGEMM_OK, GGEMM_ROUTE_EMPTY},
    {"n=0 -> EMPTY",                           P(512, 0, 512),        0, 1, EG, GGEMM_OK, GGEMM_ROUTE_EMPTY},
    {"k=0 -> K_ZERO",                          P(512, 512, 0),        0, 1, AU, GGEMM_OK, GGEMM_ROUTE_K_ZERO},
    {"big square AUTO fb=1 -> EMU",            P(8192, 8192, 8192),   0, 1, AU, GGEMM_OK, GGEMM_ROUTE_EMU},
  };
  for (const RouteCase& c : cases) {
    ggemm_config_t cfg = C(c.nm, c.fb);
    ggemm_plan_entry_t e; std::memset(&e, 0xAB, sizeof e);
    int rc = ggemm_plan(&c.p, 1, &cfg, c.path, &dm, &e);
    CHECK(rc == c.want_rc, "%s: rc=%d want %d", c.name, rc, c.want_rc);
    if (rc != GGEMM_OK || c.want_rc != GGEMM_OK) continue;
    if (c.want_route < 0)
      CHECK(e.route != GGEMM_ROUTE_SMALL_K, "%s: route=%s", c.name, route_name(e.route));
    else
      CHECK(e.route == c.want_route, "%s: route=%s want %s", c.name, route_name(e.route),
            route_name(c.want_route));
  }
  // fb=0 error must be reported even when the offending problem is not the first one.
  {
    ggemm_problem_t ps[3] = {P(128, 128, 128), P(64, 64, 64), P(128, 128, KMAX + 32)};
    ggemm_config_t cfg = C(0, 0);
    ggemm_plan_entry_t e[3];
    CHECK(ggemm_plan(ps, 3, &cfg, AU, &dm, e) == GGEMM_ERR_UNSAFE, "fb=0, unsafe problem at index 2");
  }
  // Unreachable precision target (fixed-target path): fb=0 -> error, fb=1 -> PRECISION.
  {
    ggemm_problem_t p = P(1024, 1024, 100000);
    int nm = oz2_choose_num_moduli(p.k, 1e-40);
    if (nm < 0) {
      ggemm_config_t c0 = C(0, 0, 1e-40), c1 = C(0, 1, 1e-40);
      ggemm_plan_entry_t e;
      CHECK(ggemm_plan(&p, 1, &c0, EG, &dm, &e) == GGEMM_ERR_UNSAFE, "target 1e-40 fb=0");
      CHECK(ggemm_plan(&p, 1, &c1, EG, &dm, &e) == GGEMM_OK && e.route == GGEMM_ROUTE_PRECISION,
            "target 1e-40 fb=1: route=%s", route_name(e.route));
    } else {
      std::printf("  skip PRECISION case: oz2_choose_num_moduli(100000,1e-40)=%d\n", nm);
    }
  }
}

// ================================================================= 2. num_moduli
static void test_num_moduli(const ggemm_device_model_t& dm) {
  std::printf("[num_moduli]\n");
  const int ks[] = {17, 32, 100, 512, 1024, 4096, 16384, 65536, GGEMM_MAX_K_EXACT};
  for (int k : ks) {
    ggemm_problem_t p = P(4096, 4096, k);
    ggemm_plan_entry_t e;
    ggemm_config_t ca = C(0, 0);
    CHECK(ggemm_plan(&p, 1, &ca, GGEMM_PATH_AUTO, &dm, &e) == GGEMM_OK, "k=%d", k);
    int want = oz2_choose_num_moduli_cw(k, GGEMM_RHO_BUDGET_DEFAULT);
    want = want < 2 ? 2 : (want > GGEMM_MAX_MODULI ? GGEMM_MAX_MODULI : want);
    CHECK(e.route == GGEMM_ROUTE_EMU && e.num_moduli == want,
          "AUTO adaptive k=%d: N=%d want cw()=%d route=%s", k, e.num_moduli, want, route_name(e.route));
    ggemm_config_t cf = C(7, 0);
    CHECK(ggemm_plan(&p, 1, &cf, GGEMM_PATH_AUTO, &dm, &e) == GGEMM_OK && e.num_moduli == 7,
          "AUTO fixed N=7 k=%d: N=%d", k, e.num_moduli);
    CHECK(ggemm_plan(&p, 1, &cf, GGEMM_PATH_EMU_GROUPED, &dm, &e) == GGEMM_OK && e.num_moduli == 7 &&
              e.tile_cfg == 0,
          "EMU_GROUPED fixed N=7 k=%d: N=%d tile=%d", k, e.num_moduli, e.tile_cfg);
  }
}

// ================================================================= 3. calib json
static void test_calib() {
  std::printf("[calib]\n");
  ggemm_device_model_t nom, dm;
  // nominal: no env, a GPU name that has no params file
  clean_env();
  ggemm_device_model_init(&nom, "Planner Unit Test GPU (no params file)", 9, 0, 132);
  CHECK(nom.fp64_flops > 0 && nom.int8_ops > 0 && nom.mem_bw > 0, "nominal values positive");

  // missing file -> nominal
  setenv("GGEMM_CALIB_JSON", (g_dir + "/does_not_exist.json").c_str(), 1);
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132);
  CHECK(dm.fp64_flops == nom.fp64_flops && dm.int8_ops == nom.int8_ops && dm.mem_bw == nom.mem_bw,
        "missing file -> nominal: fp64=%g/%g i8=%g/%g bw=%g/%g", dm.fp64_flops, nom.fp64_flops,
        dm.int8_ops, nom.int8_ops, dm.mem_bw, nom.mem_bw);

  // file without the fields -> nominal
  setenv("GGEMM_CALIB_JSON", (g_dir + "/no_numbers.json").c_str(), 1);
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132);
  CHECK(dm.fp64_flops == nom.fp64_flops && dm.int8_ops == nom.int8_ops && dm.mem_bw == nom.mem_bw,
        "no fields -> nominal");

  // valid file -> exactly its values
  setenv("GGEMM_CALIB_JSON", (g_dir + "/valid.json").c_str(), 1);
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132);
  CHECK(dm.fp64_flops == 5.0e13 && dm.int8_ops == 1.0e15 && dm.mem_bw == 3.0e12,
        "valid.json: fp64=%g i8=%g bw=%g", dm.fp64_flops, dm.int8_ops, dm.mem_bw);
  CHECK(dm.num_sms == 132, "num_sms=%d", dm.num_sms);

  // wrong units (67 = TFLOPS, 3.35 = TB/s): reject (value != raw) or at least warn on stderr
  setenv("GGEMM_CALIB_JSON", (g_dir + "/bad_units.json").c_str(), 1);
  std::string err = capture_stderr([&] { ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132); });
  bool rejected = dm.fp64_flops != 67.0 && dm.int8_ops != 1979.0 && dm.mem_bw != 3.35;
  CHECK(rejected || !err.empty(), "bad_units.json accepted silently: fp64=%g i8=%g bw=%g",
        dm.fp64_flops, dm.int8_ops, dm.mem_bw);
  std::printf("  bad_units: %s (fp64=%g i8=%g bw=%g)%s%s\n", rejected ? "rejected" : "ACCEPTED",
              dm.fp64_flops, dm.int8_ops, dm.mem_bw, err.empty() ? "" : " stderr: ", err.c_str());

  // individual env vars override the file
  setenv("GGEMM_CALIB_JSON", (g_dir + "/valid.json").c_str(), 1);
  setenv("GGEMM_FP64_FLOPS", "1.5e13", 1);
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132);
  CHECK(dm.fp64_flops == 1.5e13 && dm.int8_ops == 1.0e15, "env override fp64=%g", dm.fp64_flops);
  unsetenv("GGEMM_FP64_FLOPS");

  // auto-discovered bench/calib/params_<slug>.json (only if the planner was built with GGEMM_CALIB_DIR)
  unsetenv("GGEMM_CALIB_JSON");
  ggemm_device_model_init(&dm, "NVIDIA H200", 9, 0, 132);
  std::printf("  auto params (NVIDIA H200): fp64=%g i8=%g bw=%g (%s)\n", dm.fp64_flops, dm.int8_ops,
              dm.mem_bw, dm.fp64_flops == nom.fp64_flops ? "nominal" : "from file");
  CHECK(dm.fp64_flops >= 1e12 && dm.fp64_flops <= 1e15 && dm.int8_ops >= 1e13 && dm.mem_bw >= 1e11,
        "auto params out of plausible range");

  // GGEMM_TINY_MNK=0 disables the tiny route
  setenv("GGEMM_TINY_MNK", "0", 1);
  ggemm_device_model_t dz = model_h200();
  unsetenv("GGEMM_TINY_MNK");
  ggemm_device_model_t dd = model_h200();
  CHECK(dz.tiny_mnk <= 0, "GGEMM_TINY_MNK=0 -> tiny_mnk=%g (want 0)", dz.tiny_mnk);
  ggemm_problem_t tp = P(24, 24, 24);   // k > 16, m*n*k = 13824 < default 32^3
  ggemm_config_t cfg = C(0, 1);
  ggemm_plan_entry_t e;
  ggemm_plan(&tp, 1, &cfg, GGEMM_PATH_AUTO, &dd, &e);
  CHECK(e.route == GGEMM_ROUTE_TINY, "default tiny_mnk=%g: 24^3 route=%s", dd.tiny_mnk, route_name(e.route));
  ggemm_plan(&tp, 1, &cfg, GGEMM_PATH_AUTO, &dz, &e);
  CHECK(e.route != GGEMM_ROUTE_TINY, "GGEMM_TINY_MNK=0: 24^3 route=%s", route_name(e.route));
  clean_env();
}

// ================================================================= 4. tiles
static void test_tiles(const ggemm_device_model_t& dm) {
  std::printf("[tiles]\n");
  const ggemm_config_t cfg0 = C(0, 0);   // no fallback: every problem stays EMU, tile is meaningful
  // 4a. padding efficiency: many copies so the group has >> 2*SM tiles (no shrinking).
  struct TileCase { int m, n; int forbid_cfg; };   // forbid_cfg: tile that must NOT be chosen (-1 none)
  const TileCase tcs[] = {{128, 128, -1}, {256, 512, -1}, {1024, 1024, -1}, {64, 64, 0},
                          {96, 96, 0},    {32, 32, 0},    {200, 200, -1},   {1000, 40, -1}};
  for (const TileCase& tc : tcs) {
    const int G = 4000;
    std::vector<ggemm_problem_t> ps(G, P(tc.m, tc.n, 4096));
    std::vector<ggemm_plan_entry_t> e(G);
    CHECK(ggemm_plan(ps.data(), G, &cfg0, GGEMM_PATH_AUTO, &dm, e.data()) == GGEMM_OK, "plan");
    int c = e[0].tile_cfg;
    double best = 0;
    for (int t = 0; t < GGEMM_NUM_TILE_CFG; ++t) best = std::fmax(best, pad_eff(tc.m, tc.n, t));
    std::printf("  %4dx%-4d x%d -> tile %d (pad eff %.3f, best %.3f)\n", tc.m, tc.n, G,
                ggemm_tile_dim(c), pad_eff(tc.m, tc.n, c), best);
    CHECK(c >= 0 && c < GGEMM_NUM_TILE_CFG, "tile_cfg=%d out of range", c);
    CHECK(pad_eff(tc.m, tc.n, c) >= 0.75 || pad_eff(tc.m, tc.n, c) == best,
          "%dx%d: tile %d pad eff %.3f < 0.75 and not best", tc.m, tc.n, ggemm_tile_dim(c),
          pad_eff(tc.m, tc.n, c));
    if (tc.forbid_cfg >= 0) CHECK(c != tc.forbid_cfg, "%dx%d chose tile %d", tc.m, tc.n, ggemm_tile_dim(c));
    if (tc.m % 128 == 0 && tc.n % 128 == 0)
      CHECK(c == 0, "%dx%d divisible by 128 in a big group: tile %d, want 128", tc.m, tc.n, ggemm_tile_dim(c));
    bool same = true;
    for (int g = 1; g < G; ++g) same &= (e[g].tile_cfg == c);
    CHECK(same, "identical problems got different tiles");
  }
  // 4b. a group with fewer than 2*SM (prob x plane) tiles at 128x128 must shrink its tile.
  {
    struct ShrinkCase { int m, n, k, G; };
    const ShrinkCase scs[] = {{256, 256, 4096, 1}, {512, 512, 8192, 1}, {384, 384, 4096, 2}, {256, 128, 16384, 3}};
    for (const ShrinkCase& s : scs) {
      std::vector<ggemm_problem_t> ps(s.G, P(s.m, s.n, s.k));
      std::vector<ggemm_plan_entry_t> e(s.G);
      ggemm_plan(ps.data(), s.G, &cfg0, GGEMM_PATH_AUTO, &dm, e.data());
      double tiles128 = s.G * oz2_num_planes(e[0].num_moduli) * std::ceil(s.m / 128.0) * std::ceil(s.n / 128.0);
      std::printf("  shrink %dx%dx%d G=%d N=%d: 128-tiles=%g (2*SM=%d) -> tile %d\n", s.m, s.n, s.k, s.G,
                  e[0].num_moduli, tiles128, 2 * dm.num_sms, ggemm_tile_dim(e[0].tile_cfg));
      if (tiles128 < 2.0 * dm.num_sms)
        CHECK(e[0].tile_cfg > 0, "%dx%d G=%d: %g tiles < 2*SM but tile stays 128", s.m, s.n, s.G, tiles128);
    }
  }
  // 4c. route re-evaluated after tile change (AUTO, allow_fallback=1):
  //   (i) every EMU/UNPROFITABLE entry is consistent with its reported costs;
  //   (ii) the same problem planned alone (tile shrunk) vs in a big group (tile kept) must report
  //        different t_emu when the padded areas of the two final tiles differ.
  {
    const ggemm_config_t cfg1 = C(0, 1);
    std::vector<ggemm_problem_t> mix;
    const int sz[] = {24, 48, 96, 160, 224, 256, 384, 512, 768, 1024, 2048};
    const int kk[] = {17, 64, 256, 1024, 4096, 16384};
    for (int m : sz) for (int n : sz) for (int k : kk) mix.push_back(P(m, n, k, 1.0, (m + n) % 3 ? 0.0 : 1.0));
    for (int G : {1, 8, (int)mix.size()}) {
      for (int start = 0; start + G <= (int)mix.size(); start += (G == 1 ? 37 : G)) {
        std::vector<ggemm_plan_entry_t> e(G);
        ggemm_plan(mix.data() + start, G, &cfg1, GGEMM_PATH_AUTO, &dm, e.data());
        for (int g = 0; g < G; ++g) {
          const ggemm_problem_t& p = mix[start + g];
          if (e[g].route == GGEMM_ROUTE_EMU)
            CHECK(e[g].t_emu_s <= e[g].t_native_s, "EMU but t_emu %g > t_nat %g (%dx%dx%d tile %d, G=%d)",
                  e[g].t_emu_s, e[g].t_native_s, p.m, p.n, p.k, ggemm_tile_dim(e[g].tile_cfg), G);
          if (e[g].route == GGEMM_ROUTE_UNPROFITABLE)
            CHECK(e[g].t_emu_s > e[g].t_native_s, "UNPROFITABLE but t_emu %g <= t_nat %g (%dx%dx%d tile %d, G=%d)",
                  e[g].t_emu_s, e[g].t_native_s, p.m, p.n, p.k, ggemm_tile_dim(e[g].tile_cfg), G);
        }
      }
    }
    const int pm[] = {224, 200, 320, 480};
    for (int m : pm) {
      ggemm_problem_t p = P(m, m, 8192);
      ggemm_plan_entry_t alone;
      ggemm_plan(&p, 1, &cfg1, GGEMM_PATH_AUTO, &dm, &alone);
      std::vector<ggemm_problem_t> ps(4000, p);
      std::vector<ggemm_plan_entry_t> grp(4000);
      ggemm_plan(ps.data(), 4000, &cfg1, GGEMM_PATH_AUTO, &dm, grp.data());
      double ta = ggemm_tile_dim(alone.tile_cfg), tg = ggemm_tile_dim(grp[0].tile_cfg);
      bool pad_differs = std::ceil(m / ta) * ta != std::ceil(m / tg) * tg;
      std::printf("  re-eval %d^2x8192: alone tile %g %s t_emu %.3g | group tile %g %s t_emu %.3g\n", m, ta,
                  route_name(alone.route), alone.t_emu_s, tg, route_name(grp[0].route), grp[0].t_emu_s);
      if (alone.num_moduli == grp[0].num_moduli && pad_differs)
        CHECK(alone.t_emu_s != grp[0].t_emu_s, "%d^2: tile changed %g->%g but t_emu not recomputed", m, tg, ta);
    }
  }
}

// ================================================================= 5. determinism
static bool same_entry(const ggemm_plan_entry_t& a, const ggemm_plan_entry_t& b) {
  return a.route == b.route && a.num_moduli == b.num_moduli && a.tile_cfg == b.tile_cfg &&
         std::memcmp(&a.t_emu_s, &b.t_emu_s, sizeof(double)) == 0 &&
         std::memcmp(&a.t_native_s, &b.t_native_s, sizeof(double)) == 0;
}
static std::vector<ggemm_problem_t> make_group(int G, unsigned seed) {
  std::vector<ggemm_problem_t> ps(G);
  unsigned s = seed;
  auto rnd = [&s](int lo, int hi) { s = s * 1664525u + 1013904223u; return lo + (int)((s >> 8) % (unsigned)(hi - lo + 1)); };
  for (int g = 0; g < G; ++g) {
    int r = rnd(0, 99);
    int m = rnd(1, 2048), n = rnd(1, 2048), k = rnd(1, 8192);
    if (r < 3) k = 0;
    else if (r < 5) m = 0;
    else if (r < 7) k = GGEMM_MAX_K_EXACT + rnd(1, 1000);
    else if (r < 12) k = rnd(1, 16);
    ps[g] = P(m, n, k, r == 50 ? 0.0 : 1.0, (r & 1) ? 1.0 : 0.0);
  }
  return ps;
}
static void test_determinism(const ggemm_device_model_t& dm) {
  std::printf("[determinism]\n");
  for (int G : {1, 17, 1000, 65535}) {
    std::vector<ggemm_problem_t> ps = make_group(G, 12345u + G);
    for (int path : {GGEMM_PATH_AUTO, GGEMM_PATH_EMU_GROUPED}) {
      std::vector<ggemm_plan_entry_t> a(G), b(G);
      std::memset(a.data(), 0x11, G * sizeof a[0]);
      std::memset(b.data(), 0x77, G * sizeof b[0]);
      ggemm_config_t cfg = C(0, 1);
      int ra = ggemm_plan(ps.data(), G, &cfg, (ggemm_path_t)path, &dm, a.data());
      int rb = ggemm_plan(ps.data(), G, &cfg, (ggemm_path_t)path, &dm, b.data());
      int diff = 0;
      for (int g = 0; g < G; ++g) diff += !same_entry(a[g], b[g]);
      CHECK(ra == rb && ra == GGEMM_OK && diff == 0, "G=%d path=%d: rc %d/%d, %d entries differ", G, path, ra, rb, diff);
    }
  }
}

// ================================================================= 6. host time + no allocation
static void test_perf(const ggemm_device_model_t& dm) {
  std::printf("[perf]\n");
  const int G = 65535;
  std::vector<ggemm_problem_t> ps = make_group(G, 777u);
  std::vector<ggemm_plan_entry_t> e(G);
  for (int path : {GGEMM_PATH_AUTO, GGEMM_PATH_EMU_GROUPED}) {
    ggemm_config_t cfg = C(0, 1);
    g_new_calls = 0; g_count_new = true;
    ggemm_plan(ps.data(), G, &cfg, (ggemm_path_t)path, &dm, e.data());   // first call (static init)
    g_count_new = false;
    long first = g_new_calls.load();
    const int R = 5;
    double best = 1e30, sum = 0;
    g_new_calls = 0;
    for (int r = 0; r < R; ++r) {
      g_count_new = true;
      auto t0 = std::chrono::steady_clock::now();
      int rc = ggemm_plan(ps.data(), G, &cfg, (ggemm_path_t)path, &dm, e.data());
      auto t1 = std::chrono::steady_clock::now();
      g_count_new = false;
      CHECK(rc == GGEMM_OK, "rc=%d", rc);
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      best = std::fmin(best, us); sum += us;
    }
    long steady = g_new_calls.load();
    int hist[16] = {0};
    for (int g = 0; g < G; ++g) if (e[g].route >= 0 && e[g].route < 16) ++hist[e[g].route];
    std::printf("  G=%d path=%d: plan host time best %.1f us, mean %.1f us (%.2f ns/problem); "
                "operator new calls: first call %ld, steady %ld\n",
                G, path, best, sum / R, best * 1e3 / G, first, steady);
    std::printf("    routes:");
    for (int r = 0; r < 16; ++r) if (hist[r]) std::printf(" %s=%d", route_name(r), hist[r]);
    std::printf("\n");
    CHECK(steady == 0, "path=%d: %ld operator new calls inside ggemm_plan (steady state)", path, steady);
  }
}

int main(int argc, char** argv) {
  g_dir = argc > 1 ? argv[1] : "tests/testdata/planner";
  {
    std::string probe = g_dir + "/valid.json";
    FILE* f = std::fopen(probe.c_str(), "rb");
    if (!f) { std::printf("cannot open %s (pass testdata dir as argv[1])\n", probe.c_str()); return 2; }
    std::fclose(f);
  }
  clean_env();   // also GGEMM_RHO_BUDGET: ggemm_plan caches it on first call
  test_calib();
  ggemm_device_model_t dm = model_h200(132);
  std::printf("model: num_sms=%d fp64=%g i8=%g bw=%g tiny_mnk=%g\n", dm.num_sms, dm.fp64_flops,
              dm.int8_ops, dm.mem_bw, dm.tiny_mnk);
  test_routes(dm);
  test_num_moduli(dm);
  test_tiles(dm);
  test_determinism(dm);
  test_perf(dm);
  std::printf("\ntest_planner: %d passed, %d failed -> %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
