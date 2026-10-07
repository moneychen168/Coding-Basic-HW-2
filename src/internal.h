// ggemm internal contract (owner: agent-P; N/K may only ADD their own declarations).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "ggemm.h"
#include "ozaki2_numerics.h"     // agent-N (falls back to src/stubs/ when absent)
#include "int8_grouped_gemm.h"   // agent-K (falls back to src/stubs/ when absent)
#include "planner.h"

// ---- return codes of ggemm_* (public header only says "nonzero = error") ----
enum {
  GGEMM_OK = 0,
  GGEMM_ERR_INVALID = 1,        // bad args / null handle / num_problems > max_problems
  GGEMM_ERR_WORKSPACE = 2,      // call exceeds the workspace sized in ggemm_create
  GGEMM_ERR_UNSAFE = 3,         // K > GGEMM_MAX_K_EXACT and allow_fallback == 0
  GGEMM_ERR_CUDA = 4,
  GGEMM_ERR_CUBLAS = 5,
  GGEMM_ERR_NUMERICS = 6,       // oz2_* returned nonzero
  GGEMM_ERR_I8GEMM = 7,         // i8_grouped_gemm returned nonzero
  GGEMM_ERR_ALLOC = 8           // only from ggemm_create
};

// INT8 x INT8 -> INT32 exact accumulation limit. Symmetric residues of p=256 reach -128, so
// |sum| <= 128*128*kp must stay < 2^31  =>  kp <= 131040 (agent-K enforces kp <= 131040).
// NOTE: DESIGN §3 says K <= 131072; that bound overflows when a residue equals -128.
#define GGEMM_MAX_K_EXACT 131040
#define GGEMM_KP_ALIGN 32        // Kp = round_up(k, 32)
#define GGEMM_BUF_ALIGN 256      // byte alignment of every (prob, mod) block

// Tile configs understood by agent-K (DESIGN): 0=128x128, 1=64x64, 2=32x32.
#define GGEMM_NUM_TILE_CFG 3
static inline int ggemm_tile_dim(int cfg) { return cfg == 0 ? 128 : (cfg == 1 ? 64 : 32); }

static inline size_t ggemm_align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

struct ggemm_handle_s {
  cudaStream_t stream;
  int device, num_sms;
  cublasHandle_t cublas;
  void* cublas_ws; size_t cublas_ws_bytes;   // set via cublasSetWorkspace: no lazy alloc inside calls

  int max_problems;
  size_t max_A_elems, max_B_elems, max_C_elems;

  // residue / int32 workspaces (DESIGN layout [mod][prob]; per-call slab stride)
  int8_t*  A8;  size_t A8_bytes;
  int8_t*  B8;  size_t B8_bytes;
  int32_t* C32; size_t C32_bytes;
  int32_t* row_shift; size_t row_shift_elems;
  int32_t* col_shift; size_t col_shift_elems;

  // descriptors: one pinned host blob mirrored by one device blob, one H2D per call
  void* h_desc; void* d_desc; size_t desc_bytes;
  // pinned blob is double-buffered (GGEMM_DESC_SLOTS slots of desc_bytes); slot s may be
  // rewritten only after desc_done[s] (recorded at the end of the call that used it) fired.
  cudaEvent_t desc_done[2]; int desc_next;

  // agent-K LPT schedule images live inside the descriptor blob (built on host by
  // i8_grouped_gemm_build_schedule into the pinned slot, uploaded by the same single H2D).
  size_t sched_img_bytes_1;   // align(i8_grouped_gemm_sched_ws_bytes(1, 0)) for EMU_LOOP images

  // host scratch (allocated once; sized by max_problems)
  ggemm_plan_entry_t* plan;
  int* emu_idx; int* nat_idx;
  int* loop_tiles;            // [max_problems * GGEMM_MAX_MODULI] total_tiles per EMU_LOOP image
  cublasOperation_t* nat_ta; cublasOperation_t* nat_tb;
  int *nat_m, *nat_n, *nat_k, *nat_lda, *nat_ldb, *nat_ldc, *nat_gsize;
  double *nat_alpha, *nat_beta;

  ggemm_device_model_t dm;
};
