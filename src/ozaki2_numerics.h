// ozaki2_numerics.h -- Ozaki scheme II (CRT, INT8 residues): scaling/residue prep and
// CRT reconstruction for a whole group of problems.  Owner: agent-N.
//
// Layouts (see DESIGN.md):
//   A residue: per (mod l, problem) row-major m x kp  (K contiguous), base A8 + l*a8_mod_stride
//   B residue: per (mod l, problem) row-major n x kp  (= col-major K x n), base B8 + l*b8_mod_stride
//   C32:       per (mod l, problem) col-major m x n, ld = m, base C32 + l*c32_mod_stride
//   kp = round_up(k, 32); columns k..kp-1 of A8/B8 are written as 0 by oz2_prep_grouped.
//   A8/B8 bases and strides must be 4-byte aligned (int8 residues are stored as packed int32).
//
// PLANES: a problem with num_moduli = N uses N + OZ2_EXTRA_PLANES planes (l = 0..N) in A8, B8
// and C32.  Planes 0..N-1 are the CRT residues; plane N is the "abs plane" (A''=floor(2^sig|A|),
// values 0..127) whose INT8 product gives a lower bound of |A||B| for the componentwise
// certificate.  The INT8 grouped GEMM must therefore run N+1 (prob, plane) products per problem
// (all identical in shape; plane N is an ordinary int8 x int8 -> int32 GEMM).
// WORKSPACE: row_shift needs OZ2_ROW_WS_PER_ROW * m int32, col_shift OZ2_ROW_WS_PER_ROW * n.
// Reconstruct reads A, B again (only for repaired entries), so they must still be valid.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

struct oz2_problem {
  int m, n, k, kp, num_moduli, lda, ldb, ldc;
  const double *A, *B; double* C; double alpha, beta;
  int8_t *A8, *B8;               // mod 0 base; moduli separated by the strides below
  size_t a8_mod_stride, b8_mod_stride;
  int32_t* C32; size_t c32_mod_stride;
  int32_t *row_shift, *col_shift;  // workspace: 4*m and 4*n int32 (see OZ2_ROW_WS_PER_ROW)
};

#define OZ2_EXTRA_PLANES 1     // abs plane
#define OZ2_ROW_WS_PER_ROW 4   // {shift, abs shift, ||dA|| (float bits), ||A'|| (float bits)}
static inline int oz2_num_planes(int num_moduli) { return num_moduli + OZ2_EXTRA_PLANES; }

// Upload the CRT tables (moduli.h) into __constant__ memory of the current device.
// Idempotent; called lazily by the two launchers below, but should be called once from
// ggemm_create so that no H2D copy happens inside a timed region.  Returns 0 on success.
int oz2_init_constants();

// A, B scaling + residues for all problems and all moduli: exactly one kernel launch.
// d: DEVICE array of num descriptors; max_m/max_n: maxima over the group.
// Requires num <= 65535, every num_moduli in [1, GGEMM_MAX_MODULI].  Returns 0 on success.
int oz2_prep_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s);

// CRT reconstruction + inverse scaling + C = alpha*AB + beta*C: exactly one kernel launch.
// Every entry is certified against the componentwise FP64 bound (see .cu, step 6); entries that
// cannot be certified (or touch NaN/Inf) are recomputed in-kernel as FP64 dot products.
// beta == 0 does not read C.
int oz2_reconstruct_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s);

// Max k: kp * 128^2 < 2^31 for the INT32 accumulation (kp = round_up(k,32) <= 131040).
#define OZ2_MAX_K 131040
// row_shift/col_shift value marking a row/column containing NaN/Inf (entries recomputed in FP64).
#define OZ2_SHIFT_NONFINITE (-2147483647 - 1)
// abs-shift value marking an all-zero row/column (entries = beta*C exactly).
#define OZ2_SHIFT_ZERO (-2147483647)

// Host: smallest N in [2, GGEMM_MAX_MODULI] whose a-priori bound oz2_error_bound(k, N)
// <= target_rel_err; returns -1 if no N reaches the target or k > OZ2_MAX_K
// -> caller should fall back to native FP64.
int oz2_choose_num_moduli(int k, double target_rel_err);

// Host: a-priori bound on |C_ij - (AB)_ij| / (||A(i,:)||_2 ||B(:,j)||_2) for given k, N
// (alpha = 1, beta = 0).  Derivation in ozaki2_numerics.cu.
double oz2_error_bound(int k, int num_moduli);

// Host: componentwise planning.  Smallest N such that entries with
// rho_ij = ||a_i|| ||b_j|| / (|A||B|)_ij <= rho_budget are expected to pass the certificate
// (i.e. not need FP64 repair).  Never affects correctness (uncertified entries are repaired),
// only the repair rate; returns -1 only for k > OZ2_MAX_K.  Tune rho_budget with the repair
// counter below.
int oz2_choose_num_moduli_cw(int k, double rho_budget);

// Host: sticky device counters since the last reset: rows/columns with NaN/Inf seen by prep,
// and C entries recomputed in FP64 by reconstruct (certificate failed or non-finite input).
// Synchronizes s -- call outside timed regions.  Either pointer may be null.
int oz2_read_counters(int* nonfinite, unsigned long long* repaired, int reset, cudaStream_t s);
int oz2_nonfinite_count(int* count, int reset, cudaStream_t s);  // = oz2_read_counters(count, null, ...)
