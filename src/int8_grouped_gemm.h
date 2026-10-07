// src/int8_grouped_gemm.h (agent-K)
// Persistent grouped INT8 GEMM with exact INT32 accumulation.
//
// Per problem (layout fixed by DESIGN.md):
//   A : int8, m x kp, row-major (K contiguous, lda = kp)
//   B : int8, n x kp, row-major (= col-major kp x n, K contiguous, ldb = kp)
//   C : int32, m x n, col-major (ldc = m)
//   C = A * B^T  (i.e. C[i + j*m] = sum_k A[i*kp + k] * B[j*kp + k])
// Constraints: m, n >= 0 arbitrary; kp % 32 == 0 (m, n or kp == 0: problem skipped, C untouched); A/B base 16-byte aligned; C 4-byte aligned.
// Exactness: |C| <= 128*128*kp must fit int32, i.e. kp < 131072 (enforced: kp <= 131040).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

struct i8_problem {
  int m, n, kp;
  const int8_t* A;
  const int8_t* B;
  int32_t* C;
  int tile_cfg;   // 0 = 128x128, 1 = 64x64, 2 = 32x32, 3 = 32x32 split-K (2nd kernel),
                  // <0 = auto (i8_choose_tile_cfg; auto uses 3 only if enabled, see below)
};

enum {
  I8_TILE_128x128 = 0,
  I8_TILE_64x64 = 1,
  I8_TILE_32x32 = 2,
  I8_TILE_32x32_SPLITK = 3,   // intra-CTA split-K over 8 warps, runs in a separate kernel
  I8_NUM_TILE_CFGS = 4
};

// Heuristic tile config for one problem (host); returns 0..2. num_sms is currently unused.
int i8_choose_tile_cfg(int m, int n, int kp, int num_sms);
// Auto mode maps a 32x32 choice to I8_TILE_32x32_SPLITK when kp >= kp_min.
// Default: INT_MAX (never). Process-wide setting, host only.
void i8_set_auto_splitk_min_kp(int kp_min);

// Device scheduler workspace size (bytes). max_tiles is not needed by the layout (kept for API).
size_t i8_grouped_gemm_sched_ws_bytes(int max_problems, size_t max_tiles);

// Host scratch: call once at create time (e.g. in ggemm_create) with the largest problem count
// any later call will use. Pre-sizes the host-side entry/sort/staging buffers and performs the
// one-time kernel attribute + occupancy setup, so build_schedule / i8_grouped_gemm never
// allocate in the timed path. Calls beyond the reserved size still work but grow the buffers.
void i8_grouped_gemm_reserve(int max_problems, size_t max_tiles);
// Number of host-buffer growths so far (should stay constant after reserve; for tests/stats).
long long i8_grouped_gemm_host_allocs();
// Resident CTAs per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor, current device, cached)
// of the kernel that executes tile_cfg: 0/1/2 -> main kernel (same value), 3 -> split-K kernel.
// Returns -1 for an invalid tile_cfg. Grid size of a launch = SMs * this value (capped by tiles).
int i8_grouped_gemm_ctas_per_sm(int tile_cfg);

// Schedule summary. num_launches = (tiles_main > 0) + (tiles_splitk > 0), i.e. 0, 1 or 2
// kernels: the main kernel (cfg 0/1/2, unchanged round-1 SASS) and, only if the schedule has
// cfg-3 tiles, the split-K kernel. Both go on the caller's stream (split-K first).
struct i8_sched_info {
  int tiles_main;
  int tiles_splitk;
  int num_launches;
  int ws2_offset_ints;   // where image 2 starts inside the workspace (ints)
  size_t bytes_used;     // bytes of the workspace image that must be copied H2D
};

// ---- convenience path (DESIGN.md signature) --------------------------------------------
// Builds the LPT schedule on host from h_probs_for_sched (same content as d_probs), copies it
// into d_sched_ws with cudaMemcpyAsync (from a reused host staging vector -> no device or
// pinned allocation), then launches 1 or 2 persistent kernels. Returns 0 on success.
int i8_grouped_gemm(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                    void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s);
// Same, and reports the number of kernel launches issued (for ggemm_stats_t.kernel_launches).
int i8_grouped_gemm_ex(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                       void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s, int* num_launches);

// ---- split path (lets the caller put the schedule image into its own pinned buffer) -----
// New API: writes the device-workspace image into h_ws (>= i8_grouped_gemm_sched_ws_bytes(num,0));
// the caller copies info->bytes_used bytes to d_sched_ws. Supports cfg 3 (2 launches).
int i8_grouped_gemm_build_schedule(const i8_problem* h_probs, int num, void* h_ws,
                                   size_t ws_bytes, i8_sched_info* info);
// Launch only (info->num_launches kernels). The workspace self-resets its counters at the end
// of each run, so it may be re-launched without re-uploading. Returns 0 on success.
int i8_grouped_gemm_launch(const i8_problem* d_probs, int num, const void* d_sched_ws,
                           const i8_sched_info* info, cudaStream_t s);
// Number of kernels the launch above issues (0, 1 or 2) -> ggemm_stats_t.kernel_launches.
int i8_grouped_gemm_launch_count(const i8_sched_info* info);
// Same, computed from the HOST problem list (what i8_grouped_gemm / _ex will launch).
int i8_grouped_gemm_launch_count(const i8_problem* h_probs, int num);

// Legacy API (round-1 signatures, kept for source compatibility): cfg 3 is mapped to cfg 2, so
// the schedule always needs exactly 1 launch (0 if empty); *total_tiles = main-kernel tiles.
int i8_grouped_gemm_build_schedule(const i8_problem* h_probs, int num, void* h_ws,
                                   size_t ws_bytes, int* total_tiles);
int i8_grouped_gemm_launch(const i8_problem* d_probs, int num, const void* d_sched_ws,
                           int total_tiles, cudaStream_t s);

// ---- cuBLASLt INT8 reference / vendor baseline (one GEMM, same layout) ------------------
// handle: cublasLtHandle_t passed as void* to keep cublasLt.h out of this header.
// Returns 0 on success, otherwise the cublasStatus_t value (e.g. unsupported odd ldc).
int i8_gemm_cublaslt_single(void* lt_handle, int m, int n, int kp, const int8_t* A,
                            const int8_t* B, int32_t* C, void* d_workspace, size_t ws_bytes,
                            cudaStream_t s);
