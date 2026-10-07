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
  // Ablation note: every EMU_* path (2, 3, 5, 6) uses tile config 0 (128x128) for all problems
  // and never reroutes on cost; per-problem tiles and cost-based fallback exist only in AUTO.
  GGEMM_PATH_EMU_LOOP = 2,       // starter-like: single-GEMM INT8 emulation called per problem
  GGEMM_PATH_EMU_GROUPED = 3,    // ours: fused grouped pipeline (O(1) launches)
  GGEMM_PATH_AUTO = 4,           // ours + planner (per-problem moduli count, tile config, fallback)
  GGEMM_PATH_EMU_GROUPED_REUSE = 5, // EMU_GROUPED + operand reuse (prep once per unique A / B)
  GGEMM_PATH_EMU_LOOP_BATCHED = 6 // GEMMul8-like baseline: per problem 3 launches (prep, ONE
                                  // INT8 GEMM over all its planes, reconstruct)
} ggemm_path_t;

typedef struct {
  int num_moduli;        // 0 = auto (adaptive per problem); else fixed N in [2, GGEMM_MAX_MODULI]
  double target_rel_err; // used when num_moduli == 0, default 1e-15 (FP64-level)
  int allow_fallback;    // 1 = route unsafe/unprofitable problems to native FP64 (reported in stats)
  int repair_mode;       // appended: GGEMM_REPAIR_* below; 0 (also when omitted in {..}) = path default
} ggemm_config_t;

#define GGEMM_MAX_MODULI 20

// ggemm_config_t::repair_mode (componentwise FP64 certificate in reconstruct).
#define GGEMM_REPAIR_DEFAULT 0   // EMU_LOOP / EMU_LOOP_BATCHED / EMU_GROUPED / _REUSE: COUNT; AUTO: ON
#define GGEMM_REPAIR_OFF     1   // no certificate
#define GGEMM_REPAIR_COUNT   2   // certificate evaluated, failures counted, emulated value kept
#define GGEMM_REPAIR_ON      3   // failures recomputed as FP64 dot products

typedef struct {
  int num_problems;
  int num_fallback;       // problems routed to native FP64
  int kernel_launches;    // launches issued by the last call
  int moduli_min, moduli_max;
  // ---- appended (operand reuse); keep new fields at the end for ABI ----
  int reuse_enabled;      // 1 if the call ran the operand-reuse prep
  int num_unique_A;       // A operands prepped (== emulated problems when reuse is off)
  int num_unique_B;       // B operands prepped (== emulated problems when reuse is off)
  int num_promoted;       // problems run with more moduli than planned (REUSE_PROMOTE only)
  // The three counters below come from agent-N's process-wide device counters (reset before /
  // read after each call). They are NOT per handle: with several handles running concurrently on
  // different streams (or a plan executing on another stream) the counts mix and are unreliable.
  long long num_repaired; // C entries recomputed in FP64 by the certificate (repair ON);
                          // -1 = not measured (needs env GGEMM_STATS_REPAIR=1: syncs the stream)
  long long num_uncertified;      // repair COUNT: entries failing the certificate (-1 as above)
  long long num_safety_fallback;  // entries computed in FP64 due to NaN/Inf input (-1 as above)
} ggemm_stats_t;

typedef struct ggemm_handle_s* ggemm_handle_t;

// All workspace is allocated here (never inside timed calls).
// max_problems / max_total_elems bound what ggemm_grouped_dgemm may be called with.
int  ggemm_create(ggemm_handle_t* h, int max_problems, size_t max_total_A_elems,
                  size_t max_total_B_elems, size_t max_total_C_elems, cudaStream_t stream);
// Same as ggemm_create, plus min_k = smallest k of any EMULATED problem the handle will see
// (ggemm_create uses 32). Residues are stored with Kp = round_up(k, 32), so for k < 32 the
// residue workspace per element of A/B grows by up to 32/k; it is sized for
// max_{k >= min_k} round_up(k,32)/k. A call violating this returns GGEMM_ERR_WORKSPACE (2) and
// prints one diagnostic line on stderr.
int  ggemm_create_ex(ggemm_handle_t* h, int max_problems, size_t max_total_A_elems,
                     size_t max_total_B_elems, size_t max_total_C_elems, cudaStream_t stream,
                     int min_k);
void ggemm_destroy(ggemm_handle_t h);

// problems: HOST array of descriptors (pointers inside are device pointers).
// Returns 0 on success, nonzero if workspace too small / invalid args (no allocation fallback).
int ggemm_grouped_dgemm(ggemm_handle_t h, ggemm_path_t path, const ggemm_config_t* cfg,
                        const ggemm_problem_t* problems, int num_problems, ggemm_stats_t* stats);

#ifdef __cplusplus
}
#endif
