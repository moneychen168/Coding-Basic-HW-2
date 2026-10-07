// tests/test_reuse_e2e.cu -- operand reuse end to end (owner: agent-P2).
//
// GGEMM_PATH_EMU_GROUPED_REUSE vs GGEMM_PATH_EMU_GROUPED on three workloads:
//   lora : LoRA shrink, C_g = A_g (r x h) * X (h x t)    -> one shared B (X), G distinct A
//   tp   : tensor parallel, C_g = X (t x h) * W_g (h x s) -> one shared A (X), G distinct B
//   none : nothing shared (control)
// Gate: bit-exact output (memcmp) under REUSE_EXACT_N (default; same N => identical numerics),
//       and the unique-operand counts in ggemm_stats_t. AUTO (never reuses) is reported for
//       reference; EMU_LOOP_BATCHED (GEMMul8-like baseline) must also be bit-exact vs EMU_GROUPED.
// Timing (reported, not gated): cudaEvent time per call over R back-to-back calls.
// Env: GGEMM_REUSE_PROMOTE=1 (PROMOTE policy), GGEMM_STATS_REPAIR=1 (fills num_repaired).
// Usage: test_reuse_e2e [--reps R]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "ggemm.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; } else { ++g_fail; \
  printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Workload {
  std::string name;
  int G, m, n, k;
  bool share_A, share_B;
  int expect_ua, expect_ub;
  // device buffers
  double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC0 = nullptr;
  size_t nA = 0, nB = 0, nC = 0;
  std::vector<ggemm_problem_t> probs;
};

static void fill_dev(double* d, size_t n, std::mt19937_64& rng) {
  std::vector<double> h(n);
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_int_distribution<int> E(-8, 8);
  for (auto& x : h) x = std::ldexp(N(rng), E(rng));   // mixed magnitudes: exercises scaling
  CK(cudaMemcpy(d, h.data(), n * sizeof(double), cudaMemcpyHostToDevice));
}

static void setup(Workload& w, unsigned seed) {
  const size_t a1 = (size_t)w.m * w.k, b1 = (size_t)w.k * w.n, c1 = (size_t)w.m * w.n;
  w.nA = w.share_A ? a1 : a1 * w.G;
  w.nB = w.share_B ? b1 : b1 * w.G;
  w.nC = c1 * w.G;
  CK(cudaMalloc(&w.dA, w.nA * sizeof(double)));
  CK(cudaMalloc(&w.dB, w.nB * sizeof(double)));
  CK(cudaMalloc(&w.dC, w.nC * sizeof(double)));
  CK(cudaMalloc(&w.dC0, w.nC * sizeof(double)));
  std::mt19937_64 rng(seed);
  fill_dev(w.dA, w.nA, rng); fill_dev(w.dB, w.nB, rng); fill_dev(w.dC0, w.nC, rng);
  w.probs.resize(w.G);
  for (int g = 0; g < w.G; ++g) {
    ggemm_problem_t& p = w.probs[g];
    p.m = w.m; p.n = w.n; p.k = w.k;
    p.A = w.dA + (w.share_A ? 0 : (size_t)g * a1); p.lda = w.m;
    p.B = w.dB + (w.share_B ? 0 : (size_t)g * b1); p.ldb = w.k;
    p.C = w.dC + (size_t)g * c1; p.ldc = w.m;
    p.alpha = 1.0 + 0.25 * (g % 3); p.beta = (g % 2) ? 0.5 : 0.0;
  }
}

static void release(Workload& w) {
  cudaFree(w.dA); cudaFree(w.dB); cudaFree(w.dC); cudaFree(w.dC0);
}

static const char* pname(ggemm_path_t p) {
  switch (p) {
    case GGEMM_PATH_EMU_GROUPED: return "EMU_GROUPED";
    case GGEMM_PATH_EMU_GROUPED_REUSE: return "EMU_GROUPED_REUSE";
    case GGEMM_PATH_AUTO: return "AUTO";
    case GGEMM_PATH_EMU_LOOP_BATCHED: return "EMU_LOOP_BATCHED";
    default: return "?";
  }
}

static void print_stats(const char* tag, const ggemm_stats_t& s) {
  printf("  %-18s stats: probs=%d fallback=%d launches=%d moduli=[%d,%d] reuse=%d uA=%d uB=%d "
         "promoted=%d repaired=%lld\n", tag, s.num_problems, s.num_fallback, s.kernel_launches,
         s.moduli_min, s.moduli_max, s.reuse_enabled, s.num_unique_A, s.num_unique_B,
         s.num_promoted, s.num_repaired);
}

// One correctness run from C0; returns rc and copies C to host.
static int run_once(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg, Workload& w,
                    std::vector<double>& out, ggemm_stats_t* st, cudaStream_t s) {
  CK(cudaMemcpyAsync(w.dC, w.dC0, w.nC * sizeof(double), cudaMemcpyDeviceToDevice, s));
  int rc = ggemm_grouped_dgemm(h, path, cfg, w.probs.data(), w.G, st);
  CK(cudaStreamSynchronize(s));
  out.resize(w.nC);
  CK(cudaMemcpy(out.data(), w.dC, w.nC * sizeof(double), cudaMemcpyDeviceToHost));
  return rc;
}

static double time_path(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                        Workload& w, int reps, cudaStream_t s) {
  for (int i = 0; i < 3; ++i) ggemm_grouped_dgemm(h, path, cfg, w.probs.data(), w.G, nullptr);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaStreamSynchronize(s));
  CK(cudaEventRecord(e0, s));
  for (int i = 0; i < reps; ++i) ggemm_grouped_dgemm(h, path, cfg, w.probs.data(), w.G, nullptr);
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return 1e3 * ms / reps;
}

int main(int argc, char** argv) {
  int reps = 20;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);

  //               name    G    m     n     k    shareA shareB uA  uB
  std::vector<Workload> ws = {
      {"lora", 16,   16,  256, 4096, false, true,  16,  1},
      {"tp",    8,  256,  512, 4096, true,  false,  1,  8},
      {"none", 16,  256,  256, 1024, false, false, 16, 16},
  };
  size_t maxA = 0, maxB = 0, maxC = 0; int maxG = 0;
  for (size_t i = 0; i < ws.size(); ++i) {
    Workload& w = ws[i];
    setup(w, 1234 + (unsigned)i);
    // validate() counts sum(m*k) per problem (shared operands are counted once per user)
    maxA = std::max(maxA, (size_t)w.G * w.m * w.k);
    maxB = std::max(maxB, (size_t)w.G * w.k * w.n);
    maxC = std::max(maxC, w.nC);
    maxG = std::max(maxG, w.G);
  }
  cudaStream_t s; CK(cudaStreamCreate(&s));
  ggemm_handle_t h = nullptr;
  int rc = ggemm_create(&h, maxG, maxA, maxB, maxC, s);
  if (rc) { printf("ggemm_create failed: %d\n", rc); return 2; }

  ggemm_config_t cfg_emu = {0, 1e-15, 0};    // adaptive N, no fallback (pure emulation)
  ggemm_config_t cfg_auto = {0, 1e-15, 1};
  const char* promote = getenv("GGEMM_REUSE_PROMOTE");
  printf("test_reuse_e2e: reps=%d policy=%s\n", reps,
         (promote && atoi(promote) == 1) ? "PROMOTE" : "EXACT_N");

  for (Workload& w : ws) {
    printf("[%s] G=%d m=%d n=%d k=%d shareA=%d shareB=%d\n", w.name.c_str(), w.G, w.m, w.n, w.k,
           (int)w.share_A, (int)w.share_B);
    std::vector<double> c_base, c_reuse, c_auto;
    ggemm_stats_t sb, sr, sa;
    std::memset(&sb, 0, sizeof sb); std::memset(&sr, 0, sizeof sr); std::memset(&sa, 0, sizeof sa);
    int rb = run_once(h, GGEMM_PATH_EMU_GROUPED, &cfg_emu, w, c_base, &sb, s);
    int rr = run_once(h, GGEMM_PATH_EMU_GROUPED_REUSE, &cfg_emu, w, c_reuse, &sr, s);
    int ra = run_once(h, GGEMM_PATH_AUTO, &cfg_auto, w, c_auto, &sa, s);
    std::vector<double> c_lb; ggemm_stats_t sl; std::memset(&sl, 0, sizeof sl);
    int rl = run_once(h, GGEMM_PATH_EMU_LOOP_BATCHED, &cfg_emu, w, c_lb, &sl, s);
    CHECK(rl == 0, "rc loop_batched=%d", rl);
    print_stats(pname(GGEMM_PATH_EMU_LOOP_BATCHED), sl);
    CHECK(sa.reuse_enabled == 0, "AUTO must not reuse (reuse_enabled=%d)", sa.reuse_enabled);
    if (!rl && !rb) {
      CHECK(std::memcmp(c_base.data(), c_lb.data(), w.nC * sizeof(double)) == 0,
            "%s: EMU_LOOP_BATCHED != EMU_GROUPED", w.name.c_str());
      CHECK(sl.kernel_launches <= 3 * w.G, "loop_batched launches=%d > 3G", sl.kernel_launches);
    }
    CHECK(rb == 0 && rr == 0 && ra == 0, "rc base=%d reuse=%d auto=%d", rb, rr, ra);
    print_stats(pname(GGEMM_PATH_EMU_GROUPED), sb);
    print_stats(pname(GGEMM_PATH_EMU_GROUPED_REUSE), sr);
    print_stats(pname(GGEMM_PATH_AUTO), sa);
    if (rb || rr) continue;

    const bool same = std::memcmp(c_base.data(), c_reuse.data(), w.nC * sizeof(double)) == 0;
    size_t ndiff = 0;
    for (size_t i = 0; i < w.nC; ++i) ndiff += std::memcmp(&c_base[i], &c_reuse[i], sizeof(double)) != 0;
    if (sr.num_promoted == 0) CHECK(same, "%s: REUSE != EMU_GROUPED in %zu / %zu entries", w.name.c_str(), ndiff, w.nC);
    else printf("  (PROMOTE changed N for %d problems: bit-exactness not expected; %zu entries differ)\n",
                sr.num_promoted, ndiff);
    printf("  bit-exact REUSE vs EMU_GROUPED: %s (%zu differing entries)\n", same ? "yes" : "NO", ndiff);

    CHECK(sb.reuse_enabled == 0 && sr.reuse_enabled == 1, "reuse flags base=%d reuse=%d",
          sb.reuse_enabled, sr.reuse_enabled);
    CHECK(sb.num_unique_A == w.G && sb.num_unique_B == w.G, "base uA=%d uB=%d", sb.num_unique_A, sb.num_unique_B);
    CHECK(sr.num_unique_A == w.expect_ua && sr.num_unique_B == w.expect_ub,
          "%s: reuse uA=%d (exp %d) uB=%d (exp %d)", w.name.c_str(), sr.num_unique_A, w.expect_ua,
          sr.num_unique_B, w.expect_ub);
    CHECK(sr.kernel_launches == sb.kernel_launches, "launches base=%d reuse=%d",
          sb.kernel_launches, sr.kernel_launches);

    const double tb = time_path(h, GGEMM_PATH_EMU_GROUPED, &cfg_emu, w, reps, s);
    const double tr = time_path(h, GGEMM_PATH_EMU_GROUPED_REUSE, &cfg_emu, w, reps, s);
    const double ta = time_path(h, GGEMM_PATH_AUTO, &cfg_auto, w, reps, s);
    const double tl = time_path(h, GGEMM_PATH_EMU_LOOP_BATCHED, &cfg_emu, w, reps, s);
    printf("  time/call (us): EMU_LOOP_BATCHED=%.1f  EMU_GROUPED=%.1f  REUSE=%.1f  AUTO=%.1f  "
           "speedup REUSE/EMU=%.3fx\n", tl, tb, tr, ta, tb / tr);
  }

  ggemm_destroy(h);
  for (Workload& w : ws) release(w);
  cudaStreamDestroy(s);
  printf("test_reuse_e2e: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
