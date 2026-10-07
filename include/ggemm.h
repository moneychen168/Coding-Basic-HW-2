// ggemm: Grouped FP64 GEMM emulated on INT8 Tensor Cores (Ozaki scheme II / CRT).
// Column-major, no transpose (v1). C_g = alpha_g * A_g(m_g x k_g) * B_g(k_g x n_g) + beta_g * C_g
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int m, n, k;
  const double* A; int lda;   // device pointers
  const double* B; int ldb;
  double*       C; int ldc;
  double alpha, beta;
} ggemm_problem_t;

typedef enum {
  GGEMM_PATH_NATIVE_LOOP = 0,    // baseline: cublasDgemm per problem (native FP64)
  GGEMM_PATH_NATIVE_GROUPED = 1, // baseline: cublasDgemmGroupedBatched (native FP64)
  GGEMM_PATH_EMU_LOOP = 2,       // starter-like: single-GEMM INT8 emulation called per problem
  GGEMM_PATH_EMU_GROUPED = 3,    // ours: fused grouped pipeline (O(1) launches)
  GGEMM_PATH_AUTO = 4            // ours + planner (per-problem moduli count, tile config, fallback)
} ggemm_path_t;

typedef struct {
  int num_moduli;        // 0 = auto (adaptive per problem); else fixed N in [2, GGEMM_MAX_MODULI]
  double target_rel_err; // used when num_moduli == 0, default 1e-15 (FP64-level)
  int allow_fallback;    // 1 = route unsafe/unprofitable problems to native FP64 (reported in stats)
} ggemm_config_t;

#define GGEMM_MAX_MODULI 20

typedef struct {
  int num_problems;
  int num_fallback;       // problems routed to native FP64
  int kernel_launches;    // launches issued by the last call
  int moduli_min, moduli_max;
} ggemm_stats_t;

typedef struct ggemm_handle_s* ggemm_handle_t;

// All workspace is allocated here (never inside timed calls).
// max_problems / max_total_elems bound what ggemm_grouped_dgemm may be called with.
int  ggemm_create(ggemm_handle_t* h, int max_problems, size_t max_total_A_elems,
                  size_t max_total_B_elems, size_t max_total_C_elems, cudaStream_t stream);
void ggemm_destroy(ggemm_handle_t h);

// problems: HOST array of descriptors (pointers inside are device pointers).
// Returns 0 on success, nonzero if workspace too small / invalid args (no allocation fallback).
int ggemm_grouped_dgemm(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                        const ggemm_problem_t* problems, int num_problems, ggemm_stats_t* stats);

#ifdef __cplusplus
}
#endif
