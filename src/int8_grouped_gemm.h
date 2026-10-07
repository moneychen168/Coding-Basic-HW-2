// src/int8_grouped_gemm.h (agent-K)
// Persistent grouped INT8 GEMM with exact INT32 accumulation.
//
// Per problem (layout fixed by DESIGN.md):
//   A : int8, m x kp, row-major (K contiguous, lda = kp)
//   B : int8, n x kp, row-major (= col-major kp x n, K contiguous, ldb = kp)
//   C : int32, m x n, col-major (ldc = m)
//   C = A * B^T  (i.e. C[i + j*m] = sum_k A[i*kp + k] * B[j*kp + k])
// Constraints: m, n >= 1 arbitrary; kp % 32 == 0; A/B base 16-byte aligned; C 4-byte aligned.
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
  int tile_cfg;   // 0 = 128x128, 1 = 64x64, 2 = 32x32, <0 = auto (i8_choose_tile_cfg)
};

enum { I8_TILE_128x128 = 0, I8_TILE_64x64 = 1, I8_TILE_32x32 = 2, I8_NUM_TILE_CFGS = 3 };

// Heuristic tile config for one problem (host). num_sms<=0 -> query current device (cached).
int i8_choose_tile_cfg(int m, int n, int kp, int num_sms);

// Device scheduler workspace size (bytes). max_tiles is not needed by the layout (kept for API).
size_t i8_grouped_gemm_sched_ws_bytes(int max_problems, size_t max_tiles);

// ---- convenience path (DESIGN.md signature) --------------------------------------------
// Builds the LPT schedule on host from h_probs_for_sched (same content as d_probs), copies it
// into d_sched_ws with cudaMemcpyAsync (from a reused host staging vector -> no device or
// pinned allocation), then launches ONE persistent kernel. Returns 0 on success.
int i8_grouped_gemm(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                    void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s);

// ---- split path (lets the caller put the schedule image into its own pinned buffer) -----
// Writes the exact device-workspace image into h_ws (>= i8_grouped_gemm_sched_ws_bytes(num,0)).
// *total_tiles receives the number of tiles. Returns 0 on success.
int i8_grouped_gemm_build_schedule(const i8_problem* h_probs, int num, void* h_ws,
                                   size_t ws_bytes, int* total_tiles);
// Launch only. d_sched_ws must hold an image produced by build_schedule (copied by caller).
// The workspace self-resets its counter at the end of each run, so it may be re-launched.
int i8_grouped_gemm_launch(const i8_problem* d_probs, int num, const void* d_sched_ws,
                           int total_tiles, cudaStream_t s);

// ---- cuBLASLt INT8 reference / vendor baseline (one GEMM, same layout) ------------------
// handle: cublasLtHandle_t passed as void* to keep cublasLt.h out of this header.
// Returns 0 on success, otherwise the cublasStatus_t value (e.g. unsupported odd ldc).
int i8_gemm_cublaslt_single(void* lt_handle, int m, int n, int kp, const int8_t* A,
                            const int8_t* B, int32_t* C, void* d_workspace, size_t ws_bytes,
                            cudaStream_t s);
