// bench_numerics_kernels.cu (agent-N): time oz2_prep_grouped and oz2_reconstruct_grouped alone.
//
// Build:
//   /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
//     -gencode arch=compute_120,code=sm_120 -Isrc -Iinclude \
//     tests/bench_numerics_kernels.cu src/ozaki2_numerics.cu -o build/bench_numerics_kernels
// Usage: bench_numerics_kernels [--N 15] [--iters 10] [--dump file]
//   --dump writes every prep plane, the row/col workspace and the reconstructed C of every mode
//   to a binary file, so two builds (e.g. before/after a kernel change) can be compared with cmp.
//
// Metrics: median time over --iters runs (cudaEvent), and effective bandwidth over the
// COMPULSORY traffic (each byte counted once), compared with the calibrated DRAM 3.94 TB/s:
//   prep  bytes = 8 (m k + n k) [read A, B] + (N+1)(m + n) kp [write planes] + 16 (m + n) [ws]
//   recon bytes = 4 (N+1) m n [read C32] + 8 m n [write C] (+ 8 m n read C if beta != 0) + 16 (m + n)
// The INT8 products in between are produced by a deliberately simple reference kernel (untimed).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ozaki2_numerics.h"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

static const double kDramTBs = 3.94;

// planes 0..N, 16x16 tiles through shared memory (reference speed is irrelevant: untimed)
__global__ void ref_i8_gemm(const oz2_problem* probs) {
  const oz2_problem& p = probs[blockIdx.z];
  const int planes = p.num_moduli + 1;
  const int tiles_m = (p.m + 15) / 16, tiles_n = (p.n + 15) / 16;
  __shared__ int8_t As[16][33], Bs[16][33];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  for (int t = blockIdx.x; t < tiles_m * tiles_n * planes; t += gridDim.x) {
    const int l = t / (tiles_m * tiles_n), r = t % (tiles_m * tiles_n);
    const int i0 = (r % tiles_m) * 16, j0 = (r / tiles_m) * 16;
    const int8_t* A = p.A8 + l * p.a8_mod_stride;
    const int8_t* B = p.B8 + l * p.b8_mod_stride;
    int acc = 0;
    for (int k0 = 0; k0 < p.kp; k0 += 32) {
      for (int e = threadIdx.x; e < 16 * 32; e += 256) {
        const int rr = e / 32, kk = e % 32;
        As[rr][kk] = (i0 + rr < p.m) ? A[(size_t)(i0 + rr) * p.kp + k0 + kk] : 0;
        Bs[rr][kk] = (j0 + rr < p.n) ? B[(size_t)(j0 + rr) * p.kp + k0 + kk] : 0;
      }
      __syncthreads();
#pragma unroll 8
      for (int kk = 0; kk < 32; ++kk) acc += (int)As[tx][kk] * (int)Bs[ty][kk];
      __syncthreads();
    }
    if (i0 + tx < p.m && j0 + ty < p.n) p.C32[l * p.c32_mod_stride + (i0 + tx) + (size_t)(j0 + ty) * p.m] = acc;
  }
}

struct Shape { int m, n, k; };
struct Case { std::string name; std::vector<Shape> shapes; };

struct Group {
  std::vector<Shape> sh;
  int N;
  std::vector<double*> dA, dB, dC;
  int8_t *A8 = nullptr, *B8 = nullptr;
  int32_t *C32 = nullptr, *rs = nullptr, *cs = nullptr;
  size_t a8 = 0, b8 = 0, c32 = 0, rsz = 0, csz = 0;
  std::vector<oz2_problem> hp;
  oz2_problem* dp = nullptr;
  int max_m = 0, max_n = 0;
  double prep_bytes = 0, recon_bytes = 0;
};

static void build(Group& g, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> Z(0.0, 1.0);
  auto gen = [&] { return (U(rng) - 0.5) * std::exp(1.0 * Z(rng)); };
  for (auto& s : g.sh) {
    int kp = (s.k + 31) / 32 * 32, NP = oz2_num_planes(g.N);
    g.a8 += (size_t)NP * s.m * kp; g.b8 += (size_t)NP * s.n * kp; g.c32 += (size_t)NP * s.m * s.n;
    g.rsz += OZ2_ROW_WS_PER_ROW * s.m; g.csz += OZ2_ROW_WS_PER_ROW * s.n;
  }
  CK(cudaMalloc(&g.A8, g.a8 + 64)); CK(cudaMalloc(&g.B8, g.b8 + 64)); CK(cudaMalloc(&g.C32, 4 * g.c32 + 64));
  CK(cudaMalloc(&g.rs, 4 * g.rsz + 64)); CK(cudaMalloc(&g.cs, 4 * g.csz + 64));
  CK(cudaMemset(g.A8, 0, g.a8 + 64)); CK(cudaMemset(g.B8, 0, g.b8 + 64));
  CK(cudaMemset(g.rs, 0, 4 * g.rsz + 64)); CK(cudaMemset(g.cs, 0, 4 * g.csz + 64));
  size_t oa = 0, ob = 0, oc = 0, orr = 0, occ = 0;
  for (auto& s : g.sh) {
    std::vector<double> A((size_t)s.m * s.k), B((size_t)s.k * s.n);
    for (auto& x : A) x = gen();
    for (auto& x : B) x = gen();
    double *dA, *dB, *dC;
    CK(cudaMalloc(&dA, 8 * A.size() + 8)); CK(cudaMalloc(&dB, 8 * B.size() + 8)); CK(cudaMalloc(&dC, 8 * (size_t)s.m * s.n + 8));
    CK(cudaMemcpy(dA, A.data(), 8 * A.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), 8 * B.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, 8 * (size_t)s.m * s.n));
    g.dA.push_back(dA); g.dB.push_back(dB); g.dC.push_back(dC);
    oz2_problem p{};
    p.m = s.m; p.n = s.n; p.k = s.k; p.kp = (s.k + 31) / 32 * 32; p.num_moduli = g.N;
    p.lda = s.m; p.ldb = s.k; p.ldc = s.m; p.A = dA; p.B = dB; p.C = dC; p.alpha = 1.0; p.beta = 0.0;
    p.A8 = g.A8 + oa; p.a8_mod_stride = (size_t)s.m * p.kp;
    p.B8 = g.B8 + ob; p.b8_mod_stride = (size_t)s.n * p.kp;
    p.C32 = g.C32 + oc; p.c32_mod_stride = (size_t)s.m * s.n;
    p.row_shift = g.rs + orr; p.col_shift = g.cs + occ;
    const int NP = oz2_num_planes(g.N);
    oa += (size_t)NP * p.a8_mod_stride; ob += (size_t)NP * p.b8_mod_stride; oc += (size_t)NP * p.c32_mod_stride;
    orr += OZ2_ROW_WS_PER_ROW * s.m; occ += OZ2_ROW_WS_PER_ROW * s.n;
    g.hp.push_back(p);
    g.max_m = std::max(g.max_m, s.m); g.max_n = std::max(g.max_n, s.n);
    g.prep_bytes += 8.0 * ((double)s.m * s.k + (double)s.n * s.k) + (double)NP * (s.m + s.n) * p.kp + 16.0 * (s.m + s.n);
    g.recon_bytes += 4.0 * NP * (double)s.m * s.n + 8.0 * (double)s.m * s.n + 16.0 * (s.m + s.n);
  }
  CK(cudaMalloc(&g.dp, sizeof(oz2_problem) * g.hp.size()));
  CK(cudaMemcpy(g.dp, g.hp.data(), sizeof(oz2_problem) * g.hp.size(), cudaMemcpyHostToDevice));
}

static void release(Group& g) {
  for (auto p : g.dA) cudaFree(p);
  for (auto p : g.dB) cudaFree(p);
  for (auto p : g.dC) cudaFree(p);
  cudaFree(g.A8); cudaFree(g.B8); cudaFree(g.C32); cudaFree(g.rs); cudaFree(g.cs); cudaFree(g.dp);
}

static double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class F>
static double time_ms(F f, int iters) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  f(); f();  // warm-up
  CK(cudaDeviceSynchronize());
  std::vector<float> ts;
  for (int it = 0; it < iters; ++it) {
    CK(cudaEventRecord(a));
    f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return median(ts);
}

static void dump_dev(FILE* f, const void* d, size_t bytes) {
  std::vector<char> h(bytes);
  CK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
  fwrite(h.data(), 1, bytes, f);
}

int main(int argc, char** argv) {
  int N = 15, iters = 10;
  const char* dump = nullptr;
  for (int a = 1; a < argc; ++a) {
    if (!strcmp(argv[a], "--N") && a + 1 < argc) N = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--iters") && a + 1 < argc) iters = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--dump") && a + 1 < argc) dump = argv[++a];
  }
  if (oz2_init_constants()) { fprintf(stderr, "oz2_init_constants failed\n"); return 2; }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s, %d SMs, N=%d, iters=%d, DRAM ref %.2f TB/s\n", prop.name, prop.multiProcessorCount, N, iters, kDramTBs);

  std::mt19937_64 rng(2024);
  std::vector<Case> cases;
  cases.push_back({"skinny m16 n4096 k4096", {{16, 4096, 4096}}});
  cases.push_back({"skinny m16 n16 k16384", {{16, 16, 16384}}});
  cases.push_back({"skinny m4096 n16 k4096", {{4096, 16, 4096}}});
  cases.push_back({"square 1024", {{1024, 1024, 1024}}});
  cases.push_back({"square 2048", {{2048, 2048, 2048}}});
  {
    Case moe{"MoE 64 experts m~U[1,128] n2048 k1536", {}};
    std::uniform_int_distribution<int> M(1, 128);
    for (int e = 0; e < 64; ++e) moe.shapes.push_back({M(rng), 2048, 1536});
    cases.push_back(moe);
  }
  {
    Case moe{"MoE 128 experts m~U[1,32] n1536 k2048", {}};
    std::uniform_int_distribution<int> M(1, 32);
    for (int e = 0; e < 128; ++e) moe.shapes.push_back({M(rng), 1536, 2048});
    cases.push_back(moe);
  }

  FILE* df = dump ? fopen(dump, "wb") : nullptr;
  if (dump && !df) { fprintf(stderr, "cannot open %s\n", dump); return 2; }

  printf("\n%-40s %10s %9s %7s | %10s %9s %7s | %10s %10s | %9s %9s\n", "case", "prep_ms", "prep_GB/s", "%DRAM",
         "recOFF_ms", "rec_GB/s", "%DRAM", "recCNT_ms", "recON_ms", "uncert", "repaired");
  for (auto& c : cases) {
    Group g;
    g.sh = c.shapes;
    g.N = N;
    build(g, rng);
    const int num = (int)g.hp.size();
    auto prep = [&] { if (oz2_prep_grouped(g.dp, num, g.max_m, g.max_n, 0)) { fprintf(stderr, "prep failed\n"); exit(2); } };
    double t_prep = time_ms(prep, iters);
    CK(cudaDeviceSynchronize());
    // INT8 products (untimed)
    long long tiles = 0;
    for (auto& s : g.sh) tiles = std::max<long long>(tiles, (long long)((s.m + 15) / 16) * ((s.n + 15) / 16) * (N + 1));
    ref_i8_gemm<<<dim3((unsigned)std::min<long long>(tiles, 4096), 1, num), 256>>>(g.dp);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    double t_rec[3];
    unsigned long long uncert = 0, repaired = 0;
    const int modes[3] = {OZ2_REPAIR_OFF, OZ2_REPAIR_COUNT, OZ2_REPAIR_ON};
    for (int mi = 0; mi < 3; ++mi) {
      auto rec = [&] {
        if (oz2_reconstruct_grouped(g.dp, num, g.max_m, g.max_n, 0, modes[mi])) { fprintf(stderr, "recon failed\n"); exit(2); }
      };
      oz2_read_counters((oz2_counters*)nullptr, 1, 0);
      rec();  // one counted run
      oz2_counters cn{};
      oz2_read_counters(&cn, 1, 0);
      if (mi == 1) uncert = cn.uncertified;
      if (mi == 2) repaired = cn.repaired;
      if (df) {
        for (size_t p = 0; p < g.hp.size(); ++p) dump_dev(df, g.dC[p], 8 * (size_t)g.sh[p].m * g.sh[p].n);
      }
      t_rec[mi] = time_ms(rec, iters);
    }
    if (df) {
      dump_dev(df, g.A8, g.a8); dump_dev(df, g.B8, g.b8);
      dump_dev(df, g.rs, 4 * g.rsz); dump_dev(df, g.cs, 4 * g.csz);
    }
    const double gbp = g.prep_bytes / (t_prep * 1e6), gbr = g.recon_bytes / (t_rec[0] * 1e6);
    printf("%-40s %10.4f %9.1f %6.1f%% | %10.4f %9.1f %6.1f%% | %10.4f %10.4f | %9llu %9llu\n", c.name.c_str(), t_prep, gbp,
           100.0 * gbp / (kDramTBs * 1e3), t_rec[0], gbr, 100.0 * gbr / (kDramTBs * 1e3), t_rec[1], t_rec[2], uncert, repaired);
    release(g);
  }
  if (df) fclose(df);
  return 0;
}
