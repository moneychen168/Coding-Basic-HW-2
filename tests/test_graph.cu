// tests/test_graph.cu -- plan/execute (CUDA Graph) path vs plain ggemm_grouped_dgemm (owner: agent-G).
//
// Correctness (gate = bit-exact, memcmp of the FP64 output):
//   T1  plan executed on the handle stream          == direct call, same path (EMU_GROUPED, AUTO)
//   T2  plan executed on ANOTHER stream, interleaved with direct calls on the same handle that
//       use different data (shared scratch would be clobbered without the stream join)
//   T3  ggemm_plan_update_pointers to a second data set, then back
// Timing (reported, not gated), per workload, R back-to-back calls on the handle stream:
//   host_issue_us : host wall time to ISSUE one call (R calls issued, divided by R; no sync)
//   wall_us       : host wall time per call incl. final sync
//   gpu_us        : cudaEvent time per call (events on the handle stream around the R calls)
//   lat_us        : single-call latency (sync; call; sync), median of R
//   make_plan_ms  : one-off ggemm_make_plan cost
//
// Usage: test_graph [--reps R]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "ggemm.h"
#include "ggemm_graph.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...) do { if (cond) { ++g_pass; } else { ++g_fail; \
  printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct Shape { int m, n, k; };

// One data set: device A, B, C (current) and C0 (initial C), all problems packed.
struct DataSet {
  std::vector<Shape> shapes;
  double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC0 = nullptr;
  size_t nA = 0, nB = 0, nC = 0;
  std::vector<size_t> oA, oB, oC;
};

static void make_data(DataSet& d, const std::vector<Shape>& s, unsigned seed) {
  d.shapes = s;
  d.nA = d.nB = d.nC = 0;
  for (auto& x : s) {
    d.oA.push_back(d.nA); d.oB.push_back(d.nB); d.oC.push_back(d.nC);
    d.nA += (size_t)x.m * x.k; d.nB += (size_t)x.k * x.n; d.nC += (size_t)x.m * x.n;
  }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::normal_distribution<double> N(0.0, 1.0);
  auto fill = [&](size_t n) {   // mixed magnitudes so scaling / many moduli are exercised
    std::vector<double> v(n);
    for (auto& x : v) x = U(rng) * std::exp2(N(rng) * 2.0);
    return v;
  };
  std::vector<double> a = fill(d.nA), b = fill(d.nB), c = fill(d.nC);
  CK(cudaMalloc(&d.dA, std::max<size_t>(d.nA, 1) * 8));
  CK(cudaMalloc(&d.dB, std::max<size_t>(d.nB, 1) * 8));
  CK(cudaMalloc(&d.dC, std::max<size_t>(d.nC, 1) * 8));
  CK(cudaMalloc(&d.dC0, std::max<size_t>(d.nC, 1) * 8));
  CK(cudaMemcpy(d.dA, a.data(), d.nA * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.dB, b.data(), d.nB * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.dC0, c.data(), d.nC * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.dC, c.data(), d.nC * 8, cudaMemcpyHostToDevice));
}
static void free_data(DataSet& d) { cudaFree(d.dA); cudaFree(d.dB); cudaFree(d.dC); cudaFree(d.dC0); }

static std::vector<ggemm_problem_t> problems(const DataSet& d, double* C, double alpha, double beta) {
  std::vector<ggemm_problem_t> p(d.shapes.size());
  for (size_t g = 0; g < p.size(); ++g) {
    const Shape& s = d.shapes[g];
    p[g].m = s.m; p[g].n = s.n; p[g].k = s.k;
    p[g].A = d.dA + d.oA[g]; p[g].lda = std::max(1, s.m);
    p[g].B = d.dB + d.oB[g]; p[g].ldb = std::max(1, s.k);
    p[g].C = C + d.oC[g];    p[g].ldc = std::max(1, s.m);
    p[g].alpha = alpha; p[g].beta = beta;
  }
  return p;
}

static bool bit_equal(const double* d1, const double* d2, size_t n, size_t* first_diff) {
  std::vector<double> a(n), b(n);
  CK(cudaMemcpy(a.data(), d1, n * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), d2, n * 8, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i)
    if (std::memcmp(&a[i], &b[i], 8) != 0) { *first_diff = i; return false; }
  return true;
}

static const char* pname(ggemm_path_t p) { return p == GGEMM_PATH_AUTO ? "AUTO" : "EMU_GROUPED"; }

struct Workload { const char* name; std::vector<Shape> shapes; };

static std::vector<Workload> workloads() {
  std::vector<Workload> w;
  std::mt19937 r(7);
  {  // launch-overhead dominated: many tiny problems
    Workload x{"tiny64x(32,32,64)", {}};
    for (int i = 0; i < 64; ++i) x.shapes.push_back({32, 32, 64});
    w.push_back(x);
  }
  {  // MoE-like: 32 experts, ragged token counts, fixed hidden sizes
    Workload x{"moe32(m~U[16,512],n=256,k=512)", {}};
    std::uniform_int_distribution<int> M(16, 512);
    for (int i = 0; i < 32; ++i) x.shapes.push_back({M(r), 256, 512});
    w.push_back(x);
  }
  {  // mixed, incl. an empty problem, a k==0 problem and a tiny one (AUTO -> native routes)
    Workload x{"mixed16(+empty,+k0)", {}};
    std::uniform_int_distribution<int> D(8, 384);
    for (int i = 0; i < 14; ++i) x.shapes.push_back({D(r), D(r), D(r)});
    x.shapes.push_back({0, 64, 64});
    x.shapes.push_back({48, 40, 0});
    w.push_back(x);
  }
  return w;
}

int main(int argc, char** argv) {
  int reps = 200;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s (sm_%d%d)  reps=%d\n", prop.name, prop.major, prop.minor, reps);

  cudaStream_t hs, s2;
  CK(cudaStreamCreateWithFlags(&hs, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  const ggemm_config_t cfg_emu = {0, 1e-15, 0};   // adaptive moduli, FP64 target
  const ggemm_config_t cfg_auto = {0, 1e-15, 1};
  const ggemm_path_t paths[2] = {GGEMM_PATH_EMU_GROUPED, GGEMM_PATH_AUTO};

  printf("\n%-32s %-11s %8s %6s | %12s %10s %10s %10s | %12s\n", "workload", "path", "variant",
         "launch", "host_issue_us", "wall_us", "gpu_us", "lat_us", "make_plan_ms");

  for (const Workload& wl : workloads()) {
    DataSet X, Y;
    make_data(X, wl.shapes, 11);
    make_data(Y, wl.shapes, 22);   // same shapes, different data
    double *dCref, *dCg, *dCref2, *dCg2;
    CK(cudaMalloc(&dCref, std::max<size_t>(X.nC, 1) * 8)); CK(cudaMalloc(&dCg, std::max<size_t>(X.nC, 1) * 8));
    CK(cudaMalloc(&dCref2, std::max<size_t>(X.nC, 1) * 8)); CK(cudaMalloc(&dCg2, std::max<size_t>(X.nC, 1) * 8));
    const int G = (int)wl.shapes.size();
    ggemm_handle_t h = nullptr;
    int rc = ggemm_create(&h, G, 4 * X.nA + 1, 4 * X.nB + 1, X.nC + 1, hs);
    CHECK(rc == 0 && h, "%s: ggemm_create rc=%d", wl.name, rc);
    if (rc || !h) continue;

    for (ggemm_path_t path : paths) {
      const ggemm_config_t* cfg = path == GGEMM_PATH_AUTO ? &cfg_auto : &cfg_emu;
      const double alpha = 1.5, beta = 0.5;
      const size_t nC = X.nC;
      printf("[%s %s]\n", wl.name, pname(path));

      // ---- T1: direct vs plan on the handle stream
      CK(cudaMemcpy(dCref, X.dC0, nC * 8, cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(dCg, X.dC0, nC * 8, cudaMemcpyDeviceToDevice));
      auto pr = problems(X, dCref, alpha, beta);
      auto pg = problems(X, dCg, alpha, beta);
      ggemm_stats_t st_d, st_p;
      rc = ggemm_grouped_dgemm(h, path, cfg, pr.data(), G, &st_d);
      CK(cudaStreamSynchronize(hs));
      CHECK(rc == 0, "direct rc=%d", rc);
      int perr = 0;
      ggemm_plan_t plan = ggemm_make_plan_ex(h, path, cfg, pg.data(), G, &perr);
      CHECK(plan != nullptr, "make_plan err=%d", perr);
      if (!plan) continue;
      ggemm_plan_get_stats(plan, &st_p);
      rc = ggemm_execute_plan(plan, hs);
      CK(cudaStreamSynchronize(hs));
      size_t fd = 0;
      bool eq = rc == 0 && bit_equal(dCref, dCg, nC, &fd);
      CHECK(eq, "T1 bit-exact (rc=%d first diff at %zu)", rc, fd);
      CHECK(st_p.num_fallback == st_d.num_fallback && st_p.moduli_min == st_d.moduli_min &&
            st_p.moduli_max == st_d.moduli_max,
            "stats differ: fallback %d/%d moduli %d-%d / %d-%d", st_d.num_fallback, st_p.num_fallback,
            st_d.moduli_min, st_d.moduli_max, st_p.moduli_min, st_p.moduli_max);
      printf("  T1 handle-stream    : %s  (launches direct=%d graph=%d, fallback=%d, moduli %d-%d)\n",
             eq ? "bit-exact" : "MISMATCH", st_d.kernel_launches, st_p.kernel_launches,
             st_p.num_fallback, st_p.moduli_min, st_p.moduli_max);

      // ---- T2: plan on s2 interleaved with direct calls (other data) on the handle stream
      auto py = problems(Y, Y.dC, alpha, 0.0);
      for (int it = 0; it < 8; ++it) {
        CK(cudaMemcpyAsync(dCg, X.dC0, nC * 8, cudaMemcpyDeviceToDevice, s2));
        rc |= ggemm_execute_plan(plan, s2);
        rc |= ggemm_grouped_dgemm(h, path, cfg, py.data(), G, nullptr);
      }
      CK(cudaStreamSynchronize(s2)); CK(cudaStreamSynchronize(hs));
      eq = rc == 0 && bit_equal(dCref, dCg, nC, &fd);
      CHECK(eq, "T2 bit-exact (rc=%d first diff at %zu)", rc, fd);
      printf("  T2 other stream     : %s\n", eq ? "bit-exact" : "MISMATCH");

      // ---- T3: update_pointers -> Y data (C into dCg2), compare with direct on Y; then back to X
      {
        CK(cudaMemcpy(dCref2, Y.dC0, nC * 8, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dCg2, Y.dC0, nC * 8, cudaMemcpyDeviceToDevice));
        auto pyr = problems(Y, dCref2, alpha, beta);
        auto pyg = problems(Y, dCg2, alpha, beta);
        rc = ggemm_grouped_dgemm(h, path, cfg, pyr.data(), G, nullptr);
        std::vector<const double*> A(G), B(G); std::vector<double*> C(G);
        for (int g = 0; g < G; ++g) { A[g] = pyg[g].A; B[g] = pyg[g].B; C[g] = pyg[g].C; }
        rc |= ggemm_plan_update_pointers(plan, A.data(), B.data(), C.data());
        rc |= ggemm_execute_plan(plan, s2);   // also exercises update -> other-stream ordering
        CK(cudaStreamSynchronize(s2)); CK(cudaStreamSynchronize(hs));
        eq = rc == 0 && bit_equal(dCref2, dCg2, nC, &fd);
        CHECK(eq, "T3a update bit-exact (rc=%d first diff at %zu)", rc, fd);
        // back to X (only C pointer array given => A/B also need X: pass all three)
        for (int g = 0; g < G; ++g) { A[g] = pg[g].A; B[g] = pg[g].B; C[g] = pg[g].C; }
        CK(cudaMemcpy(dCg, X.dC0, nC * 8, cudaMemcpyDeviceToDevice));
        rc = ggemm_plan_update_pointers(plan, A.data(), B.data(), C.data());
        rc |= ggemm_execute_plan(plan, hs);
        CK(cudaStreamSynchronize(hs));
        bool eq2 = rc == 0 && bit_equal(dCref, dCg, nC, &fd);
        CHECK(eq2, "T3b update-back bit-exact (rc=%d first diff at %zu)", rc, fd);
        printf("  T3 update_pointers  : %s / %s\n", eq ? "bit-exact" : "MISMATCH",
               eq2 ? "bit-exact" : "MISMATCH");
      }
      ggemm_destroy_plan(plan);

      // ---- timing (beta = 0 so repeated calls stay finite; output not checked here)
      auto pt = problems(X, dCg, 1.0, 0.0);
      auto t_mk = clk::now();
      ggemm_plan_t tp = ggemm_make_plan(h, path, cfg, pt.data(), G);
      const double make_ms = us_since(t_mk) / 1e3;
      CHECK(tp != nullptr, "timing make_plan");
      if (!tp) continue;
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      for (int variant = 0; variant < 2; ++variant) {
        auto call = [&]() {
          return variant == 0 ? ggemm_grouped_dgemm(h, path, cfg, pt.data(), G, nullptr)
                              : ggemm_execute_plan(tp, hs);
        };
        for (int i = 0; i < 10; ++i) rc |= call();   // warm-up
        CK(cudaStreamSynchronize(hs));
        // throughput: R back-to-back calls
        CK(cudaEventRecord(e0, hs));
        auto t0 = clk::now();
        for (int i = 0; i < reps; ++i) rc |= call();
        const double issue_us = us_since(t0) / reps;
        CK(cudaEventRecord(e1, hs));
        CK(cudaStreamSynchronize(hs));
        const double wall_us = us_since(t0) / reps;
        float gpu_ms = 0; CK(cudaEventElapsedTime(&gpu_ms, e0, e1));
        // latency: one call at a time
        std::vector<double> lat(reps);
        for (int i = 0; i < reps; ++i) {
          auto t1 = clk::now();
          rc |= call();
          CK(cudaStreamSynchronize(hs));
          lat[i] = us_since(t1);
        }
        std::nth_element(lat.begin(), lat.begin() + reps / 2, lat.end());
        CHECK(rc == 0, "timing rc=%d", rc);
        printf("%-32s %-11s %8s %6d | %12.2f %10.2f %10.2f %10.2f | %12.3f\n", wl.name, pname(path),
               variant == 0 ? "direct" : "graph", variant == 0 ? st_d.kernel_launches : st_p.kernel_launches,
               issue_us, wall_us, 1e3 * gpu_ms / reps, lat[reps / 2], variant == 1 ? make_ms : 0.0);
      }
      cudaEventDestroy(e0); cudaEventDestroy(e1);
      ggemm_destroy_plan(tp);
    }
    ggemm_destroy(h);
    cudaFree(dCref); cudaFree(dCg); cudaFree(dCref2); cudaFree(dCg2);
    free_data(X); free_data(Y);
  }
  cudaStreamDestroy(hs); cudaStreamDestroy(s2);
  printf("\ntest_graph: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
