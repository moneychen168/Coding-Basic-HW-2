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

// ---- NVTX ranges (header-only nvtx3). -DGGEMM_NVTX=0 compiles them out (empty macros).
//   GGEMM_NVTX_RANGE(var, "name")  opens a range closed at scope exit (RAII) or by
//   GGEMM_NVTX_END(var) (idempotent), whichever comes first.
#ifndef GGEMM_NVTX
#define GGEMM_NVTX 1
#endif
#if GGEMM_NVTX
#include <nvtx3/nvToolsExt.h>
struct ggemm_nvtx_range {
  bool on;
  explicit ggemm_nvtx_range(const char* s) : on(true) { nvtxRangePushA(s); }
  void end() { if (on) { nvtxRangePop(); on = false; } }
  ~ggemm_nvtx_range() { end(); }
  ggemm_nvtx_range(const ggemm_nvtx_range&) = delete;
  ggemm_nvtx_range& operator=(const ggemm_nvtx_range&) = delete;
};
#define GGEMM_NVTX_RANGE(var, name) ggemm_nvtx_range var(name)
#define GGEMM_NVTX_END(var) (var).end()
#else
#define GGEMM_NVTX_RANGE(var, name)
#define GGEMM_NVTX_END(var)
#endif

struct reuse_arena;   // src/operand_reuse.h (agent-O): host pass, shared A/B operand detection

// Per-call descriptor blob layout (identical in the pinned slot and in the device blob):
//   [oz2 recon x num_emu] [oz2 prep x num_prep] [i8_problem x num_i8]
//   [A ptrs | B ptrs | C ptrs  x num_nat] [LPT schedule image(s)]   (image last: partial H2D)
// num_prep == 0 (no operand reuse): prep uses the recon descriptors (off_prep == off_oz).
// num_prep  > 0 (operand reuse): UA A-only (n = 0) then UB B-only (m = 0) descriptors.
struct ggemm_blob_layout { size_t off_oz, off_prep, off_i8, off_sched, off_ptr, total; };
ggemm_blob_layout ggemm_layout_for(size_t num_emu, size_t num_prep, size_t num_i8,
                                   size_t sched_bytes, size_t num_nat);
// cfg->repair_mode (GGEMM_REPAIR_*, 0 = path default) -> OZ2_REPAIR_*; -1 if out of range.
int ggemm_oz2_repair_mode(ggemm_path_t path, const ggemm_config_t* cfg);
// Stats counters (env GGEMM_STATS_REPAIR=1 only): reset is async (enqueued on s); read syncs s
// and fills num_repaired / num_uncertified / num_safety_fallback (all -1 when disabled).
int ggemm_counters_reset(ggemm_handle_t h, cudaStream_t s);
int ggemm_counters_read(ggemm_handle_t h, cudaStream_t s, ggemm_stats_t* st);
// Native FP64 grouping (shared with the graph module): host arrays of >= cnt entries each.
struct ggemm_nat_groups {
  cublasOperation_t *ta, *tb;
  int *m, *n, *k, *lda, *ldb, *ldc, *gsize;
  double *alpha, *beta;
};
// Sorts idx[0..cnt) so equal (m,n,k,lda,ldb,ldc,alpha,beta) problems are contiguous, fills one
// entry per group in out, returns the group count. Pointer arrays must follow the sorted idx.
int ggemm_group_native(const ggemm_problem_t* probs, int* idx, int cnt, const ggemm_nat_groups& out);
// Argument / element-budget check shared by ggemm_grouped_dgemm and the graph module.
int ggemm_validate(ggemm_handle_t h, const ggemm_problem_t* p, int num);

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
  size_t sched_img_bytes_P;   // align(sched_ws_bytes(max planes, 0)): one EMU_LOOP_BATCHED image

  // host scratch (allocated once; sized by max_problems)
  ggemm_plan_entry_t* plan;
  int* plan_memo; size_t plan_memo_cap;   // ggemm_plan_ex shape memo (>= 2 max_problems, pow2)
  int* emu_idx; int* nat_idx;
  i8_sched_info* loop_info;   // [max_problems * planes] agent-K schedule info per EMU_LOOP(_BATCHED) image
  cublasOperation_t* nat_ta; cublasOperation_t* nat_tb;
  int *nat_m, *nat_n, *nat_k, *nat_lda, *nat_ldb, *nat_ldc, *nat_gsize;
  double *nat_alpha, *nat_beta;

  ggemm_device_model_t dm;

  // operand reuse (GGEMM_PATH_EMU_GROUPED_REUSE only; AUTO never reuses).
  // Arena reserved at create (max_problems) -> build_operand_reuse never allocates.
  reuse_arena* reuse;
  int reuse_promote;          // env GGEMM_REUSE_PROMOTE=1 -> REUSE_PROMOTE, else REUSE_EXACT_N
  int* reuse_nmod;            // [max_problems] N_g input of build_operand_reuse (0 = not emulated)
  size_t *ua_off8, *ua_offs;  // [max_problems] unique A: A8 offset (bytes), row_shift offset (int32)
  size_t *ub_off8, *ub_offs;  // [max_problems] unique B: B8 offset (bytes), col_shift offset (int32)
  int min_k;                  // ggemm_create_ex: residue workspace sized for k >= min_k
  int ws_diag_printed;        // one-shot stderr diagnostic for GGEMM_ERR_WORKSPACE
  int stats_repair;           // env GGEMM_STATS_REPAIR=1: fill stats.num_repaired (synchronizes)
};
