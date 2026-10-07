// tests/reference.h -- double-double GEMM reference + error metrics (owner: agent-B).
// Used only for verification; never inside a timed region.
//
// Metrics (all per problem g, then max over g):
//   err_cw  = max_ij |C_ij - Cref_ij| / den_ij,
//             den_ij = |alpha| (|A||B|)_ij + |beta| |C0_ij|      (beta==0 -> C0 term omitted)
//             -> the quantity bounded by the classical FP64 result
//                |C - fl(C)| <= gamma_{k+2} * den (gamma_k if alpha=1, beta=0),  gamma_n = n u / (1 - n u),  u = 2^-53.
//   rel_max = max_ij |C_ij - Cref_ij| / |Cref_ij|   (over Cref_ij != 0; informational only,
//             unbounded under cancellation, so never used as a gate)
//   ratio_vs_native = err_cw(path) / err_cw(cuBLAS DGEMM on the same data)
// Cref is alpha*A*B + beta*C0 accumulated in double-double (~2^-104 relative to den), so
// its own error is ~10^-15 x the FP64 bound and negligible.
#pragma once
#include <cublas_v2.h>
#include <vector>
#include "../bench/workloads.h"

struct RefBuffers {
  double *hi = nullptr, *lo = nullptr, *den = nullptr;
  unsigned long long* d_stats = nullptr;   // 3 per problem: err_cw bits, rel bits, nonfinite count
  size_t n = 0; int G = 0;
};

struct ErrStats {
  double err_cw = 0;       // max componentwise error (normalised by den)
  double rel_max = 0;      // max relative error
  double ratio_gamma = 0;  // max_g err_cw_g / gamma_{n_g}, n_g = fp64_bound_terms  (<= 1 means FP64-level bound met)
  int worst_problem = -1;  // argmax of err_cw_g / gamma
  unsigned long long nonfinite = 0;
  std::vector<double> per_problem_cw;   // err_cw_g
};

double fp64_gamma(long long n);   // n u / (1 - n u)
// Rounding steps in fl(alpha*fl(A*B) + beta*C0): k for the dot product, +1 if alpha != 1,
// +1 if beta != 0 (Higham, Accuracy & Stability, 2nd ed., Sec. 3.5). The FP64 contract is
// |C - Cref| <= gamma_{n} * den componentwise with n = fp64_bound_terms(p).
long long fp64_bound_terms(const ggemm_problem_t& p);

int  ref_alloc(RefBuffers* r, const WorkloadBuffers& b);
void ref_free(RefBuffers* r);
// Cref <- alpha*A*B + beta*C0 in double-double, den as above. Reads b.dA, b.dB, b.dC0.
int  ref_compute(const WorkloadBuffers& b, RefBuffers* r);
// Compare b.dC against the reference.
int  ref_compare(const WorkloadBuffers& b, RefBuffers* r, ErrStats* out);
// Independent native baseline: cublasDgemm per problem on b.dC (call reset_C first).
int  run_cublas_dgemm_loop(const WorkloadBuffers& b, cublasHandle_t h);
