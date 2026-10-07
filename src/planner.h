// Planner (agent-P): per-problem emulate/fallback, num_moduli, tile config. Host-only.
#pragma once
#include "ggemm.h"

// Why a problem was routed to native FP64 (0 = emulated).
typedef enum {
  GGEMM_ROUTE_EMU = 0,
  GGEMM_ROUTE_EMPTY = 1,        // m==0 || n==0 : nothing to do
  GGEMM_ROUTE_K_ZERO = 2,       // k==0 : C = beta*C, done by cuBLAS
  GGEMM_ROUTE_K_TOO_LARGE = 3,  // k > GGEMM_MAX_K_EXACT (INT32 overflow)
  GGEMM_ROUTE_TINY = 4,         // m*n*k < tiny_mnk
  GGEMM_ROUTE_UNPROFITABLE = 5, // cost model: emulation slower than native
  GGEMM_ROUTE_PRECISION = 6,    // oz2_choose_num_moduli: no N <= GGEMM_MAX_MODULI meets target
  GGEMM_ROUTE_SMALL_K = 7,      // AUTO: k <= GGEMM_AUTO_NATIVE_MAX_K -> native FP64
  GGEMM_ROUTE_ALPHA_ZERO = 8    // alpha == 0: C = beta*C, done by cuBLAS (all emulation paths)
} ggemm_route_t;

// AUTO only (with allow_fallback): k at or below this goes straight to native FP64.
#define GGEMM_AUTO_NATIVE_MAX_K 16
// AUTO adaptive N uses oz2_choose_num_moduli_cw(k, rho_budget). rho_budget = env
// GGEMM_RHO_BUDGET, else this default. PLACEHOLDER, NOT TUNED: tune it with
// stats.num_repaired (GGEMM_STATS_REPAIR=1); it only moves the FP64-repair rate, never accuracy.
#define GGEMM_RHO_BUDGET_DEFAULT 16.0

// Native routes that are exact by construction (not "fallbacks" in ggemm_stats_t).
static inline int ggemm_route_is_fallback(int route) {
  return route != GGEMM_ROUTE_EMU && route != GGEMM_ROUTE_EMPTY && route != GGEMM_ROUTE_K_ZERO &&
         route != GGEMM_ROUTE_ALPHA_ZERO;
}

typedef struct {
  int route;        // ggemm_route_t
  int num_moduli;   // valid when route == EMU
  int tile_cfg;     // 0=128x128, 1=64x64, 2=32x32
  double t_emu_s, t_native_s;  // cost-model estimates (AUTO only; 0 otherwise)
} ggemm_plan_entry_t;

// Throughput model (same terms as bench/calib/fit.py). Source, in priority order:
//   1. individual env vars GGEMM_FP64_FLOPS / GGEMM_INT8_OPS / GGEMM_MEM_BW / GGEMM_TINY_MNK
//   2. env GGEMM_CALIB_JSON, else bench/calib/params_<gpu_slug>.json (measured, auto)
//   3. nominal spec-sheet values per arch (NOT measured; placeholders only).
typedef struct {
  int num_sms;
  double fp64_flops;     // achievable native DGEMM FLOP/s
  double int8_ops;       // achievable INT8 grouped GEMM OP/s (agent-K kernel)
  double mem_bw;         // achievable DRAM bytes/s for prep / reconstruct
  double tiny_mnk;       // m*n*k below this -> native
} ggemm_device_model_t;

#ifdef __cplusplus
extern "C" {
#endif

void ggemm_device_model_init(ggemm_device_model_t* dm, const char* gpu_name, int cc_major,
                             int cc_minor, int num_sms);

// Number of moduli for one problem honoring cfg (fixed N, else oz2_choose_num_moduli).
// Returns -1 when the adaptive choice cannot meet target_rel_err.
int ggemm_plan_num_moduli(const ggemm_config_t* cfg, int k);

// path in {EMU_LOOP, EMU_GROUPED, AUTO}. Fills out[0..num).
//  - EMU_LOOP / EMU_GROUPED: emulate everything; tile_cfg = 0; only safety
//    reroutes (empty, k==0, k too large if allow_fallback) — never cost-based.
//  - AUTO: plus tiny / cost-model fallback (if allow_fallback) and per-problem tile_cfg.
// Returns 0, or GGEMM_ERR_UNSAFE if (k > GGEMM_MAX_K_EXACT or target unreachable) and
// allow_fallback == 0.
int ggemm_plan(const ggemm_problem_t* probs, int num, const ggemm_config_t* cfg,
               ggemm_path_t path, const ggemm_device_model_t* dm, ggemm_plan_entry_t* out);

#ifdef __cplusplus
}
#endif
