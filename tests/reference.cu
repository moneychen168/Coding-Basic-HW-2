// tests/reference.cu -- GPU double-double reference GEMM and error metrics (owner: agent-B).
#include "reference.h"
#include <cmath>
#include <cstdio>
#include <cstring>

#define R_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

long long fp64_bound_terms(const ggemm_problem_t& p) {
  long long n = (long long)p.k + (p.alpha != 1.0 ? 1 : 0) + (p.beta != 0.0 ? 1 : 0);
  return n < 1 ? 1 : n;
}
double fp64_gamma(long long n) {
  const double u = 0x1.0p-53;
  return (double)n * u / (1.0 - (double)n * u);
}

// --------------------------------------------------------------- dd arithmetic
struct dd { double hi, lo; };
__device__ __forceinline__ dd two_sum(double a, double b) {
  double s = a + b, bb = s - a; return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ dd quick_two_sum(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
__device__ __forceinline__ dd dd_add(dd a, dd b) {          // accurate (IEEE-style) dd add
  dd s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
  s.lo += t.hi; s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo; return quick_two_sum(s.hi, s.lo);
}
__device__ __forceinline__ dd dd_mul_d(dd a, double b) {
  double p = a.hi * b, e = fma(a.hi, b, -p);
  e = fma(a.lo, b, e); return quick_two_sum(p, e);
}
__device__ __forceinline__ dd two_prod(double a, double b) { double p = a * b; return {p, fma(a, b, -p)}; }

constexpr int T = 16;
__global__ void k_ref_dd(ggemm_problem_t p, const double* C0, double* hi, double* lo, double* den) {
  __shared__ double As[T][T + 1], Bs[T][T + 1];
  int i = blockIdx.x * T + threadIdx.x, j = blockIdx.y * T + threadIdx.y;
  dd acc = {0.0, 0.0};
  double aacc = 0.0;
  for (int k0 = 0; k0 < p.k; k0 += T) {
    // A tile: rows blockIdx.x*T.., cols k0..; B tile: rows k0.., cols blockIdx.y*T..
    int ar = blockIdx.x * T + threadIdx.x, ac = k0 + threadIdx.y;
    As[threadIdx.x][threadIdx.y] = (ar < p.m && ac < p.k) ? p.A[ar + (size_t)ac * p.lda] : 0.0;
    int br = k0 + threadIdx.x, bc = blockIdx.y * T + threadIdx.y;
    Bs[threadIdx.x][threadIdx.y] = (br < p.k && bc < p.n) ? p.B[br + (size_t)bc * p.ldb] : 0.0;
    __syncthreads();
#pragma unroll 4
    for (int kk = 0; kk < T; ++kk) {
      double a = As[threadIdx.x][kk], b = Bs[kk][threadIdx.y];
      acc = dd_add(acc, two_prod(a, b));
      aacc = fma(fabs(a), fabs(b), aacc);
    }
    __syncthreads();
  }
  if (i >= p.m || j >= p.n) return;
  size_t idx = i + (size_t)j * p.ldc;
  dd r = dd_mul_d(acc, p.alpha);
  double d = fabs(p.alpha) * aacc;
  if (p.beta != 0.0) {
    double c0 = C0[idx];
    r = dd_add(r, two_prod(p.beta, c0));
    d += fabs(p.beta) * fabs(c0);
  }
  hi[idx] = r.hi; lo[idx] = r.lo; den[idx] = d;
}

__global__ void k_cmp(int g, int m, int n, int ldc, const double* C, const double* hi, const double* lo,
                      const double* den, unsigned long long* st) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (i >= m || j >= n) return;
  size_t idx = i + (size_t)j * ldc;
  double c = C[idx];
  if (!isfinite(c)) { atomicAdd(&st[3 * g + 2], 1ull); return; }
  double e = fabs((c - hi[idx]) - lo[idx]);
  double ref = hi[idx] + lo[idx];
  double cw = (den[idx] > 0) ? e / den[idx] : (e == 0 ? 0.0 : INFINITY);
  double rl = (ref != 0) ? e / fabs(ref) : 0.0;
  // non-negative doubles order like their bit patterns
  atomicMax(&st[3 * g + 0], (unsigned long long)__double_as_longlong(cw));
  atomicMax(&st[3 * g + 1], (unsigned long long)__double_as_longlong(rl));
}

int ref_alloc(RefBuffers* r, const WorkloadBuffers& b) {
  *r = RefBuffers();
  r->n = b.nC; r->G = (int)b.probs.size();
  R_CUDA(cudaMalloc(&r->hi, (b.nC + 1) * sizeof(double)));
  R_CUDA(cudaMalloc(&r->lo, (b.nC + 1) * sizeof(double)));
  R_CUDA(cudaMalloc(&r->den, (b.nC + 1) * sizeof(double)));
  R_CUDA(cudaMalloc(&r->d_stats, (3 * (size_t)r->G + 1) * sizeof(unsigned long long)));
  return 0;
}
void ref_free(RefBuffers* r) {
  cudaFree(r->hi); cudaFree(r->lo); cudaFree(r->den); cudaFree(r->d_stats); *r = RefBuffers();
}

int ref_compute(const WorkloadBuffers& b, RefBuffers* r) {
  for (size_t g = 0; g < b.probs.size(); ++g) {
    const ggemm_problem_t& p = b.probs[g];
    if (p.m == 0 || p.n == 0) continue;
    size_t o = b.offC[g];
    dim3 grid((p.m + T - 1) / T, (p.n + T - 1) / T), blk(T, T);
    k_ref_dd<<<grid, blk>>>(p, b.dC0 + o, r->hi + o, r->lo + o, r->den + o);
  }
  R_CUDA(cudaGetLastError());
  R_CUDA(cudaDeviceSynchronize());
  return 0;
}

int ref_compare(const WorkloadBuffers& b, RefBuffers* r, ErrStats* out) {
  int G = (int)b.probs.size();
  R_CUDA(cudaMemset(r->d_stats, 0, (3 * (size_t)G + 1) * sizeof(unsigned long long)));
  for (int g = 0; g < G; ++g) {
    const ggemm_problem_t& p = b.probs[g];
    if (p.m == 0 || p.n == 0) continue;
    size_t o = b.offC[g];
    dim3 grid((p.m + 127) / 128, p.n);
    k_cmp<<<grid, 128>>>(g, p.m, p.n, p.ldc, b.dC + o, r->hi + o, r->lo + o, r->den + o, r->d_stats);
  }
  R_CUDA(cudaGetLastError());
  std::vector<unsigned long long> st(3 * (size_t)G + 1);
  R_CUDA(cudaMemcpy(st.data(), r->d_stats, st.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  *out = ErrStats();
  out->per_problem_cw.resize(G);
  for (int g = 0; g < G; ++g) {
    double cw, rl;
    memcpy(&cw, &st[3 * g], 8); memcpy(&rl, &st[3 * g + 1], 8);
    if (st[3 * g + 2]) { cw = INFINITY; out->nonfinite += st[3 * g + 2]; }
    out->per_problem_cw[g] = cw;
    if (cw > out->err_cw) out->err_cw = cw;
    if (rl > out->rel_max) out->rel_max = rl;
    const ggemm_problem_t& p = b.probs[g];
    if (p.m == 0 || p.n == 0) continue;
    double rg = cw / fp64_gamma(fp64_bound_terms(p));
    if (!(rg <= out->ratio_gamma)) { out->ratio_gamma = rg; out->worst_problem = g; }
  }
  return 0;
}

int run_cublas_dgemm_loop(const WorkloadBuffers& b, cublasHandle_t h) {
  for (auto& p : b.probs) {
    if (p.m == 0 || p.n == 0) continue;
    cublasStatus_t s = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, p.m, p.n, p.k, &p.alpha, p.A, p.lda,
                                   p.B, p.ldb, &p.beta, p.C, p.ldc);
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasDgemm failed %d\n", (int)s); return 1; }
  }
  R_CUDA(cudaDeviceSynchronize());
  return 0;
}
