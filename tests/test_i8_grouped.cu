// tests/test_i8_grouped.cu (agent-K): correctness (bit-exact vs CPU int64 reference and vs
// cuBLASLt) + TOPS of the persistent grouped INT8 GEMM.
// Build (login node: compile only):
//   /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
//     -gencode arch=compute_120,code=sm_120 -Isrc tests/test_i8_grouped.cu src/int8_grouped_gemm.cu \
//     -lcublasLt -o build/test_i8_grouped
// Run (compute node only): ./build/test_i8_grouped [--quick | --perf] [--iters N]
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "int8_grouped_gemm.h"

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__,          \
                   cudaGetErrorString(e_));                                            \
      std::exit(2);                                                                    \
    }                                                                                  \
  } while (0)

struct Shape { int m, n, kp; };

struct Group {
  std::vector<Shape> shapes;
  std::vector<int8_t> hA, hB;            // host copies
  std::vector<size_t> offA, offB, offC;  // element offsets
  size_t totA = 0, totB = 0, totC = 0;
  int8_t *dA = nullptr, *dB = nullptr;
  int32_t *dC = nullptr, *dRef = nullptr;
  std::vector<i8_problem> hp;
  i8_problem* dp = nullptr;
  double ops = 0;  // 2*m*n*kp summed
};

static size_t up(size_t x, size_t a) { return (x + a - 1) / a * a; }

static void build_group(Group& g, const std::vector<Shape>& shapes, uint32_t seed, int fill_mode) {
  g.shapes = shapes;
  for (auto& s : shapes) {
    g.offA.push_back(g.totA); g.totA += up((size_t)s.m * s.kp, 256);
    g.offB.push_back(g.totB); g.totB += up((size_t)s.n * s.kp, 256);
    g.offC.push_back(g.totC); g.totC += up((size_t)s.m * s.n, 64);
    g.ops += 2.0 * s.m * s.n * (double)s.kp;
  }
  g.hA.assign(g.totA, 0); g.hB.assign(g.totB, 0);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(-128, 127);
  for (size_t i = 0; i < g.totA; ++i) g.hA[i] = (int8_t)(fill_mode ? -128 : d(rng));
  for (size_t i = 0; i < g.totB; ++i) g.hB[i] = (int8_t)(fill_mode ? -128 : d(rng));
  CK(cudaMalloc(&g.dA, g.totA)); CK(cudaMalloc(&g.dB, g.totB));
  CK(cudaMalloc(&g.dC, g.totC * 4)); CK(cudaMalloc(&g.dRef, g.totC * 4));
  CK(cudaMemcpy(g.dA, g.hA.data(), g.totA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g.dB, g.hB.data(), g.totB, cudaMemcpyHostToDevice));
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto& s = shapes[i];
    g.hp.push_back({s.m, s.n, s.kp, g.dA + g.offA[i], g.dB + g.offB[i], g.dC + g.offC[i], -1});
  }
  CK(cudaMalloc(&g.dp, sizeof(i8_problem) * shapes.size()));
}

static void upload_probs(Group& g, int cfg) {
  for (auto& p : g.hp) p.tile_cfg = cfg;
  CK(cudaMemcpy(g.dp, g.hp.data(), sizeof(i8_problem) * g.hp.size(), cudaMemcpyHostToDevice));
}

static void free_group(Group& g) {
  cudaFree(g.dA); cudaFree(g.dB); cudaFree(g.dC); cudaFree(g.dRef); cudaFree(g.dp);
}

static inline int64_t ref_elem(const Group& g, size_t p, int i, int j) {
  const Shape& s = g.shapes[p];
  const int8_t* a = g.hA.data() + g.offA[p] + (size_t)i * s.kp;
  const int8_t* b = g.hB.data() + g.offB[p] + (size_t)j * s.kp;
  int64_t acc = 0;
  for (int k = 0; k < s.kp; ++k) acc += (int32_t)a[k] * (int32_t)b[k];
  return acc;
}

// Compare device result (g.dC) with CPU. full: every element; else `samples` random + corners.
static long long check_cpu(const Group& g, bool full, int samples, const char* tag) {
  std::vector<int32_t> hC(g.totC);
  CK(cudaMemcpy(hC.data(), g.dC, g.totC * 4, cudaMemcpyDeviceToHost));
  long long bad = 0;
  std::mt19937 rng(7);
  for (size_t p = 0; p < g.shapes.size(); ++p) {
    const Shape& s = g.shapes[p];
    if (s.kp == 0 || s.m == 0 || s.n == 0) continue;   // skipped by design (C untouched)
    const int32_t* c = hC.data() + g.offC[p];
    auto chk = [&](int i, int j) {
      int64_t r = ref_elem(g, p, i, j);
      if ((int64_t)c[(size_t)j * s.m + i] != r) {
        if (bad < 10)
          std::printf("  [%s] MISMATCH prob %zu (%d,%d,%d) at (%d,%d): got %d ref %lld\n", tag, p,
                      s.m, s.n, s.kp, i, j, c[(size_t)j * s.m + i], (long long)r);
        ++bad;
      }
    };
    if (full) {
      for (int j = 0; j < s.n; ++j)
        for (int i = 0; i < s.m; ++i) chk(i, j);
    } else {
      chk(0, 0); chk(s.m - 1, s.n - 1); chk(s.m - 1, 0); chk(0, s.n - 1);
      for (int t = 0; t < samples; ++t) chk(rng() % s.m, rng() % s.n);
    }
  }
  return bad;
}

// cuBLASLt into dRef for every problem; returns number of problems cuBLASLt refused.
static int run_lt(cublasLtHandle_t lt, Group& g, void* ws, size_t wsb, cudaStream_t st,
                  std::vector<char>* ok) {
  int refused = 0;
  if (ok) ok->assign(g.shapes.size(), 0);
  for (size_t p = 0; p < g.shapes.size(); ++p) {
    auto& s = g.shapes[p];
    if (s.kp == 0 || s.m == 0 || s.n == 0) continue;   // not a GEMM; ours skips it
    int rc = i8_gemm_cublaslt_single(lt, s.m, s.n, s.kp, g.hp[p].A, g.hp[p].B, g.dRef + g.offC[p],
                                     ws, wsb, st);
    if (rc) ++refused;
    else if (ok) (*ok)[p] = 1;
  }
  return refused;
}

static long long check_vs_lt(const Group& g, const std::vector<char>& ok) {
  std::vector<int32_t> a(g.totC), b(g.totC);
  CK(cudaMemcpy(a.data(), g.dC, g.totC * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), g.dRef, g.totC * 4, cudaMemcpyDeviceToHost));
  long long bad = 0;
  for (size_t p = 0; p < g.shapes.size(); ++p) {
    if (!ok[p]) continue;
    size_t n = (size_t)g.shapes[p].m * g.shapes[p].n;
    for (size_t i = 0; i < n; ++i) bad += a[g.offC[p] + i] != b[g.offC[p] + i];
  }
  return bad;
}

static int g_fail = 0;
static void report(const char* what, long long bad) {
  std::printf("%-58s %s (%lld mismatches)\n", what, bad ? "FAIL" : "PASS", bad);
  if (bad) g_fail = 1;
}

int main(int argc, char** argv) {
  // --quick: correctness only; --perf: performance only; --iters N: timed iterations (default 20)
  bool quick = false, perf_only = false;
  int iters = 20;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--quick")) quick = true;
    else if (!std::strcmp(argv[a], "--perf")) perf_only = true;
    else if (!std::strcmp(argv[a], "--iters") && a + 1 < argc) iters = std::atoi(argv[++a]);
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cublasLtHandle_t lt;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasLtCreate failed\n"); return 2; }
  size_t lt_wsb = 64ull << 20;
  void* lt_ws;
  CK(cudaMalloc(&lt_ws, lt_wsb));
  const int max_probs = 4096;
  size_t sched_b = i8_grouped_gemm_sched_ws_bytes(max_probs, 0);
  void* d_sched;
  CK(cudaMalloc(&d_sched, sched_b));
  CK(cudaMemset(d_sched, 0, sched_b));
  void* h_sched;
  CK(cudaMallocHost(&h_sched, sched_b));

  // ---------------- 1. heterogeneous correctness, full CPU check, every tile config ----------
  if (!perf_only) {
    std::vector<Shape> shapes = {
        {1, 1, 32},      {1, 257, 4096},  {257, 1, 96},    {33, 17, 64},    {129, 131, 1024},
        {127, 255, 32},  {64, 64, 65536}, {3, 5, 131040},  {500, 300, 2048}, {1000, 1, 512},
        {1, 1000, 512},  {77, 1000, 320}, {1024, 768, 1024}, {31, 33, 160},  {0, 7, 64},
        {65, 63, 8192},  {129, 1, 131040}, {2, 2, 224},   {9, 7, 0}};
    Group g;
    build_group(g, shapes, 1234u, 0);
    const char* names[4] = {"cfg0 128x128", "cfg1 64x64", "cfg2 32x32", "auto"};
    for (int c = 0; c < 4; ++c) {
      int cfg = c < 3 ? c : -1;
      upload_probs(g, cfg);
      CK(cudaMemset(g.dC, 0x7f, g.totC * 4));
      int rc = i8_grouped_gemm(g.dp, (int)g.hp.size(), g.hp.data(), d_sched, sched_b, st);
      CK(cudaStreamSynchronize(st));
      CK(cudaGetLastError());
      if (rc) { std::printf("i8_grouped_gemm rc=%d\n", rc); g_fail = 1; }
      std::string t = std::string("hetero 19 probs, full CPU bit-exact, ") + names[c];
      report(t.c_str(), check_cpu(g, true, 0, names[c]));
    }
    // Relaunch via split path twice (tests counter self-reset), auto cfg.
    {
      int total = 0;
      int rc = i8_grouped_gemm_build_schedule(g.hp.data(), (int)g.hp.size(), h_sched, sched_b, &total);
      CK(cudaMemcpyAsync(d_sched, h_sched, sched_b, cudaMemcpyHostToDevice, st));
      for (int r = 0; r < 2; ++r) {
        CK(cudaMemsetAsync(g.dC, 0, g.totC * 4, st));
        rc |= i8_grouped_gemm_launch(g.dp, (int)g.hp.size(), d_sched, total, st);
      }
      CK(cudaStreamSynchronize(st));
      if (rc) { std::printf("split path rc=%d\n", rc); g_fail = 1; }
      report("split path, 2nd launch w/o re-upload (counter reset)", check_cpu(g, true, 0, "split"));
    }
    // Extreme values: all -128 -> |C| = 16384*kp, kp=131040 is the int32 edge.
    {
      Group e;
      build_group(e, {{3, 5, 131040}, {17, 9, 131040}, {1, 1, 32}}, 1u, 1);
      upload_probs(e, -1);
      int rc = i8_grouped_gemm(e.dp, (int)e.hp.size(), e.hp.data(), d_sched, sched_b, st);
      CK(cudaStreamSynchronize(st));
      if (rc) g_fail = 1;
      report("all -128 inputs, kp=131040 (int32 edge)", check_cpu(e, true, 0, "edge"));
      free_group(e);
    }
    std::vector<char> ok;
    int refused = run_lt(lt, g, lt_ws, lt_wsb, st, &ok);
    CK(cudaStreamSynchronize(st));
    std::printf("cuBLASLt refused %d / %zu hetero problems (empty ones not submitted)\n", refused, g.shapes.size());
    report("cuBLASLt vs ours (auto), supported problems", check_vs_lt(g, ok));
    free_group(g);
  }
  if (quick) { std::printf(g_fail ? "SOME TESTS FAILED\n" : "ALL PASS\n"); return g_fail; }

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const int warm = 3;
  auto time_ms = [&](auto&& fn) {
    for (int i = 0; i < warm; ++i) fn();
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) fn();
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return (double)ms / iters;
  };

  // ---------------- 2. performance sets ----------------
  struct Set { const char* name; std::vector<Shape> shapes; };
  std::vector<Set> sets;
  {
    std::mt19937 r(42);
    std::vector<Shape> v;
    for (int i = 0; i < 64; ++i) {
      int m = 1 + r() % 2048, n = 1 + r() % 2048, kp = 32 * (1 + r() % 128);
      v.push_back({m, n, kp});
    }
    sets.push_back({"random64 (m,n<=2048, kp<=4096)", v});
  }
  sets.push_back({"ozaki-like 2048^3 x 14 moduli", std::vector<Shape>(14, {2048, 2048, 2048})});
  {
    std::vector<Shape> v;
    for (int i = 0; i < 256; ++i) v.push_back({1 + (i * 37) % 200, 1 + (i * 53) % 200, 32 * (1 + i % 16)});
    sets.push_back({"small256 (m,n<=200, kp<=512)", v});
  }
  sets.push_back({"single 8192^3", {{8192, 8192, 8192}}});
  sets.push_back({"single 4096x4096x16384", {{4096, 4096, 16384}}});

  std::printf("\n%-34s %-12s %10s %10s %9s\n", "set", "variant", "ms", "TOPS", "check");
  for (auto& S : sets) {
    Group g;
    build_group(g, S.shapes, 99u, 0);
    std::vector<char> ok;
    run_lt(lt, g, lt_ws, lt_wsb, st, &ok);   // reference into dRef
    CK(cudaStreamSynchronize(st));
    const int ncfg = 4;
    const char* names[ncfg] = {"auto", "cfg0", "cfg1", "cfg2"};
    for (int c = 0; c < ncfg; ++c) {
      int cfg = c == 0 ? -1 : c - 1;
      upload_probs(g, cfg);
      int np = (int)g.hp.size();
      double ms = time_ms([&] { i8_grouped_gemm(g.dp, np, g.hp.data(), d_sched, sched_b, st); });
      long long bad = check_cpu(g, false, 256, names[c]) + check_vs_lt(g, ok);
      if (bad) g_fail = 1;
      std::printf("%-34s %-12s %10.4f %10.2f %9s\n", S.name, names[c], ms, g.ops / ms / 1e9,
                  bad ? "FAIL" : "ok");
      if (c == 0) {   // launch-only (schedule pre-uploaded from pinned buffer)
        int total = 0;
        i8_grouped_gemm_build_schedule(g.hp.data(), np, h_sched, sched_b, &total);
        CK(cudaMemcpyAsync(d_sched, h_sched, sched_b, cudaMemcpyHostToDevice, st));
        double ms2 = time_ms([&] { i8_grouped_gemm_launch(g.dp, np, d_sched, total, st); });
        std::printf("%-34s %-12s %10.4f %10.2f %9s\n", S.name, "auto-launch", ms2,
                    g.ops / ms2 / 1e9, "-");
      }
    }
    int refused = 0;
    double msl = time_ms([&] { refused = run_lt(lt, g, lt_ws, lt_wsb, st, nullptr); });
    std::printf("%-34s %-12s %10.4f %10.2f %9s\n", S.name, "cuBLASLt-loop", msl, g.ops / msl / 1e9,
                refused ? "refused" : "-");
    free_group(g);
  }
  std::printf("TOPS = 2*m*n*kp / time. RTX 5090 dense INT8 peak = 838 TOPS (see src comment).\n");
  std::printf(g_fail ? "SOME TESTS FAILED\n" : "ALL PASS\n");
  return g_fail;
}
