// src/int8_grouped_gemm.cu (agent-K)
// Persistent grouped INT8 GEMM: mma.sync m16n8k32 s8.s8.s32, cp.async multi-stage pipeline,
// ldmatrix with XOR-swizzled shared memory, exact INT32 accumulation. No CUTLASS.
//
// Peak reference (for TOPS %-of-peak in reports): RTX 5090 (GB202, 170 SMs, 2.407 GHz boost)
// dense INT8 tensor = 838 TOPS, sparse (2:4) = 1676 TOPS per NVIDIA's Blackwell (RTX) whitepaper;
// cross-check: 170 SM * 2.407e9 Hz * 2048 int8 ops/clk/SM = 837.9e12. NOTE: some third-party
// spec sites label 838 as the *sparse* figure; the arithmetic above (same 2048 ops/clk/SM as
// Ada, whose 4090 dense INT8 = 660.6 TOPS) supports 838 = dense. Real clocks under load vary.
//
// ------------------------------------------------------------------------------------------
// Design summary
//  * ONE kernel launch for the whole problem list. All three tile configs live in the same
//    kernel (switch on the per-tile cfg). Reason: (a) DESIGN.md wants the grouped pipeline to be a
//    fixed 3-kernel chain, (b) a single global work queue lets LPT balance tiles of *different*
//    configs against each other, so there is one tail instead of one tail per config launch.
//    Cost: every config shares the 256-thread block and 48 KB dynamic smem budget and the max
//    register count. We make the configs "shape-compatible": each one moves exactly
//    512 x 16 B chunks of A and of B per pipeline stage (2 cp.async per thread per operand), so
//    smaller tiles get a proportionally deeper BK instead of wasting the smem budget:
//        cfg0 128x128, BK=64,  3 stages  (warp tile 64x32, 16 mma / k32-step)
//        cfg1  64x64,  BK=128, 3 stages  (warp tile 32x16,  4 mma / k32-step)
//        cfg2  32x32,  BK=256, 3 stages  (warp tile 16x8,   1 mma / k32-step)
//    (Round 2 tried thread-0 decode overlap, cross-tile prologue overlap, no zero-fill and an
//    intra-CTA split-K cfg2; H200 ablation job 510965 showed a structural ~10% cfg0 regression,
//    so this file is round 1 + the safe fixes. Round-2 source: build/abl/round2_int8_grouped_gemm.cu.)
//    Split-K 32x32 (tile_cfg 3) lives in a SEPARATE kernel (main-kernel SASS == round 1, verified
//    with cuobjdump). Decision (H200 job 511054, 2 passes, ms): auto-sk stays OFF by default
//    (g_auto_splitk_min_kp = INT_MAX), no kp_min threshold is set:
//      workload (auto-sk launches)   auto            auto-sk          net
//      small256 (2)                  0.0236/0.0236   0.0310/0.0313    -31% (slower)
//      mix 8x2048^3+64 skinny (2)    0.3373/0.3373   0.3481/0.3476    -3%  (stable across passes)
//      random64, ozaki (1)           identical within 0.3% (auto never picks 32x32 there)
//    cfg3 alone beats cfg2 (forced, 1 launch): small256 0.0432 vs 0.0455 ms, mix 190 vs 135
//    TOPS, 8192^3 211 vs 162 TOPS -- but in auto the 32x32 tiles are a minority, and moving them
//    to a 2nd serialized kernel loses the tail filling they do inside the main kernel and adds a
//    launch, which outweighs the faster tile at both kp<=512 and kp=4096. cfg3 remains available
//    by explicit tile_cfg = 3 (or i8_set_auto_splitk_min_kp). Possible future rule (unmeasured in
//    auto mode): use cfg3 only when ALL tiles of a call are 32x32 (still 1 launch).
//    Noise reference from the same job: cfg0 vs round-1 binary (identical SASS) differs <=0.2% on
//    grouped sets and up to 2.1% on single-GEMM sets; auto vs auto-sk on single 4096 (identical
//    schedule) differs up to 5%, so single-GEMM rows need >5% to be meaningful.
//    k32-steps beyond kp inside a BK tile are skipped (CTA-uniform branch), so the deep BK of
//    small configs does not waste MMAs when kp is small.
//    48 KB smem + <=128 regs -> 2 CTAs/SM on both sm_90 (228 KB smem) and sm_120 (~100 KB smem).
//  * Scheduler: host sorts problems by per-tile cost (LPT, heaviest tiles first) and builds the
//    tile prefix sum; persistent CTAs pull tile ids from an atomic counter (prefetched one tile
//    ahead) and binary-search the prefix table. Zero-tile problems are dropped from the table.
//    The workspace self-resets (last CTA zeroes the counter), and the host image also carries
//    counter = 0, so no memset launch is needed.
//  * Bounds: rows >= m / n and k-chunks >= kp are zero-filled by cp.async src-size = 0;
//    the epilogue store is predicated. kp % 32 == 0 means every 16 B chunk is fully in or out.
//  * Bank conflicts: smem row = BK bytes, 16 B chunk index XOR-swizzled with the row:
//      BK >= 128 (>= 8 chunks/row): chunk ^ (row & 7)
//      BK == 64  (4 chunks/row, two rows per 128 B): chunk ^ ((row >> 1) & 3)
//    so each 8-address ldmatrix phase (8 consecutive rows, same logical chunk) hits 8 distinct
//    16 B bank groups.
// ------------------------------------------------------------------------------------------
#include "int8_grouped_gemm.h"

#include <cublasLt.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr int kThreads = 256;           // 8 warps, arranged 2 (M) x 4 (N)
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kStages = 3;
constexpr int kSmemBytes = 48 * 1024;   // (BM+BN)*BK*kStages for every config
constexpr int kHdrInts = 8;             // ws header: counter, done, total_tiles, num_entries, pad

template <int BM_, int BN_, int BK_>
struct TileCfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WTM = BM / kWarpsM, WTN = BN / kWarpsN;
  static constexpr int MF = WTM / 16, NF = WTN / 8;
  static constexpr int CPR = BK / 16;                    // 16 B chunks per smem row
  static constexpr int A_CHUNKS = BM * CPR, B_CHUNKS = BN * CPR;
  static constexpr int STAGE_BYTES = (BM + BN) * BK;
  static_assert(A_CHUNKS % kThreads == 0 && B_CHUNKS % kThreads == 0, "chunk split");
  static_assert(STAGE_BYTES * kStages <= kSmemBytes, "smem budget");
  static_assert(MF >= 1 && NF >= 1 && (NF == 1 || NF % 2 == 0), "warp tile");
  static_assert(CPR == 4 || CPR >= 8, "swizzle");
};
using Cfg0 = TileCfg<128, 128, 64>;
using Cfg1 = TileCfg<64, 64, 128>;
using Cfg2 = TileCfg<32, 32, 256>;

__host__ __device__ constexpr int cfg_bm(int c) { return c == 0 ? 128 : (c == 1 ? 64 : 32); }

template <int CPR>
__device__ __forceinline__ int swz(int row, int chunk) {
  if constexpr (CPR == 4) return chunk ^ ((row >> 1) & 3);
  else return chunk ^ (row & 7);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                       uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Issue the cp.async for one (BM x BK) A tile and one (BN x BK) B tile into a stage.
template <class C>
__device__ __forceinline__ void load_stage(int8_t* stage, const int8_t* A, const int8_t* B,
                                           int m_rem, int n_rem, int kp, int k0, int tid) {
  const uint32_t sA = smem_u32(stage);
  const uint32_t sB = sA + C::BM * C::BK;
#pragma unroll
  for (int i = 0; i < C::A_CHUNKS / kThreads; ++i) {
    int c = tid + i * kThreads;
    int row = c / C::CPR, ch = c % C::CPR;
    int kk = k0 + ch * 16;
    bool p = (row < m_rem) && (kk < kp);
    const int8_t* src = p ? (A + (size_t)row * kp + kk) : A;
    cp_async16(sA + row * C::BK + swz<C::CPR>(row, ch) * 16, src, p);
  }
#pragma unroll
  for (int i = 0; i < C::B_CHUNKS / kThreads; ++i) {
    int c = tid + i * kThreads;
    int row = c / C::CPR, ch = c % C::CPR;
    int kk = k0 + ch * 16;
    bool p = (row < n_rem) && (kk < kp);
    const int8_t* src = p ? (B + (size_t)row * kp + kk) : B;
    cp_async16(sB + row * C::BK + swz<C::CPR>(row, ch) * 16, src, p);
  }
}

template <class C>
__device__ __forceinline__ void run_tile(int8_t* smem, const i8_problem& P, int tm, int tn) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / kWarpsN, wn = warp % kWarpsN;
  const int m0 = tm * C::BM, n0 = tn * C::BN;
  const int m_rem = P.m - m0, n_rem = P.n - n0, kp = P.kp;
  const int8_t* A = P.A + (size_t)m0 * kp;
  const int8_t* B = P.B + (size_t)n0 * kp;
  const int ktiles = (kp + C::BK - 1) / C::BK;

  int acc[C::MF][C::NF][4];
#pragma unroll
  for (int i = 0; i < C::MF; ++i)
#pragma unroll
    for (int j = 0; j < C::NF; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < ktiles) load_stage<C>(smem + s * C::STAGE_BYTES, A, B, m_rem, n_rem, kp, s * C::BK, tid);
    cp_async_commit();
  }

  // Per-lane ldmatrix row/chunk offsets (independent of k).
  const int a_row_base = wm * C::WTM + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_chunk_off = lane >> 4;
  const int b_row_base = wn * C::WTN + (lane & 7) + ((lane >> 4) & 1) * 8;
  const int b_chunk_off = (lane >> 3) & 1;

  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      int nk = kt + kStages - 1;
      if (nk < ktiles)
        load_stage<C>(smem + (nk % kStages) * C::STAGE_BYTES, A, B, m_rem, n_rem, kp, nk * C::BK, tid);
      cp_async_commit();
    }
    const uint32_t sA = smem_u32(smem + (kt % kStages) * C::STAGE_BYTES);
    const uint32_t sB = sA + C::BM * C::BK;
    const int k0 = kt * C::BK;
#pragma unroll
    for (int ks = 0; ks < C::BK / 32; ++ks) {
      if (k0 + ks * 32 < kp) {   // CTA-uniform: skip all-zero k32 steps of the last BK tile
        uint32_t af[C::MF][4];
        uint32_t bf[C::NF][2];
#pragma unroll
        for (int i = 0; i < C::MF; ++i) {
          int r = a_row_base + i * 16;
          int ch = ks * 2 + a_chunk_off;
          ldsm_x4(sA + r * C::BK + swz<C::CPR>(r, ch) * 16, af[i][0], af[i][1], af[i][2], af[i][3]);
        }
        if constexpr (C::NF == 1) {
          int r = wn * C::WTN + (lane & 7);
          int ch = ks * 2 + ((lane >> 3) & 1);
          ldsm_x2(sB + r * C::BK + swz<C::CPR>(r, ch) * 16, bf[0][0], bf[0][1]);
        } else {
#pragma unroll
          for (int j = 0; j < C::NF / 2; ++j) {
            int r = b_row_base + j * 16;
            int ch = ks * 2 + b_chunk_off;
            ldsm_x4(sB + r * C::BK + swz<C::CPR>(r, ch) * 16, bf[2 * j][0], bf[2 * j][1],
                    bf[2 * j + 1][0], bf[2 * j + 1][1]);
          }
        }
#pragma unroll
        for (int i = 0; i < C::MF; ++i)
#pragma unroll
          for (int j = 0; j < C::NF; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();   // smem is reused by the next tile's prologue

  // Epilogue: col-major C (ld = m). For a fixed column, lanes g=0..7 write 8 consecutive rows
  // (32 B segment) -> full-sector stores.
  const int g = lane >> 2, t = lane & 3;
  int32_t* Cp = P.C;
  const int M = P.m;
#pragma unroll
  for (int i = 0; i < C::MF; ++i) {
    int r0 = m0 + wm * C::WTM + i * 16 + g;
#pragma unroll
    for (int j = 0; j < C::NF; ++j) {
      int c0 = n0 + wn * C::WTN + j * 8 + t * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int r = r0 + h * 8;
        if (r < M) {
          if (c0 < P.n) Cp[(size_t)c0 * M + r] = acc[i][j][2 * h];
          if (c0 + 1 < P.n) Cp[(size_t)(c0 + 1) * M + r] = acc[i][j][2 * h + 1];
        }
      }
    }
  }
}

// cfg2 (32x32, BK=256) with intra-CTA split-K. Touches ONLY the cfg2 path: it reuses
// load_stage<Cfg2> (same smem layout / swizzle / zero-fill) and the same prologue/pipeline
// structure as run_tile, but all 8 warps own the full 32x32 tile and warp w computes k32-step w
// of every BK=256 stage (2x4 m16n8 fragments = 8 mma per 4 ldmatrix.x4). Partial sums are
// reduced through shared memory (one int4 per (warp, fragment, lane); 8*8*32*16 B = 32 KB).
__device__ __forceinline__ void run_tile_cfg2_splitk(int8_t* smem, const i8_problem& P, int tm,
                                                     int tn) {
  using C = Cfg2;
  constexpr int MF = C::BM / 16, NF = C::BN / 8;          // 2 x 4 fragments per warp
  static_assert(C::BK / 32 == kThreads / 32, "one k32 step per warp per stage");
  static_assert(MF * NF * 32 == kThreads, "one reduction slot per thread");
  static_assert((kThreads / 32) * MF * NF * 32 * 16 <= kSmemBytes, "reduction buffer");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = tm * C::BM, n0 = tn * C::BN;
  const int m_rem = P.m - m0, n_rem = P.n - n0, kp = P.kp;
  const int8_t* A = P.A + (size_t)m0 * kp;
  const int8_t* B = P.B + (size_t)n0 * kp;
  const int ktiles = (kp + C::BK - 1) / C::BK;

  int acc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < ktiles) load_stage<C>(smem + s * C::STAGE_BYTES, A, B, m_rem, n_rem, kp, s * C::BK, tid);
    cp_async_commit();
  }
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_ch = warp * 2 + (lane >> 4);
  const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8, b_ch = warp * 2 + ((lane >> 3) & 1);

  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      int nk = kt + kStages - 1;
      if (nk < ktiles)
        load_stage<C>(smem + (nk % kStages) * C::STAGE_BYTES, A, B, m_rem, n_rem, kp, nk * C::BK, tid);
      cp_async_commit();
    }
    if (kt * C::BK + warp * 32 < kp) {   // warp-uniform: this warp's k32 step is inside kp
      const uint32_t sA = smem_u32(smem + (kt % kStages) * C::STAGE_BYTES);
      const uint32_t sB = sA + C::BM * C::BK;
      uint32_t af[MF][4], bf[NF][2];
#pragma unroll
      for (int i = 0; i < MF; ++i) {
        int r = a_row + i * 16;
        ldsm_x4(sA + r * C::BK + swz<C::CPR>(r, a_ch) * 16, af[i][0], af[i][1], af[i][2], af[i][3]);
      }
#pragma unroll
      for (int j = 0; j < NF / 2; ++j) {
        int r = b_row + j * 16;
        ldsm_x4(sB + r * C::BK + swz<C::CPR>(r, b_ch) * 16, bf[2 * j][0], bf[2 * j][1],
                bf[2 * j + 1][0], bf[2 * j + 1][1]);
      }
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();   // all stages consumed -> smem reused as the reduction buffer

  int4* red = reinterpret_cast<int4*>(smem);
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
      red[warp * kThreads + (i * NF + j) * 32 + lane] =
          make_int4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
  __syncthreads();
  // Thread tid owns slot tid = (fragment f, lane). In-range sums are exact (|C| fits int32);
  // unsigned adds keep rows/cols beyond m/n (zero-filled anyway) free of signed-overflow UB.
  unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
  for (int q = 0; q < kThreads / 32; ++q) {
    int4 x = red[q * kThreads + tid];
    s0 += (unsigned)x.x; s1 += (unsigned)x.y; s2 += (unsigned)x.z; s3 += (unsigned)x.w;
  }
  // The kernel loop's __syncthreads after run_tile protects `red` from the next prologue.
  const int f = tid >> 5, fi = f / NF, fj = f % NF;
  const int g = lane >> 2, t = lane & 3;
  const int r0 = m0 + fi * 16 + g, c0 = n0 + fj * 8 + t * 2;
  const int v[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
  int32_t* Cp = P.C;
  const int M = P.m;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int r = r0 + h * 8;
    if (r < M) {
      if (c0 < P.n) Cp[(size_t)c0 * M + r] = v[2 * h];
      if (c0 + 1 < P.n) Cp[(size_t)(c0 + 1) * M + r] = v[2 * h + 1];
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2)
    i8_grouped_persistent_kernel(const i8_problem* __restrict__ probs, int* __restrict__ ws,
                                 int total_tiles) {
  extern __shared__ __align__(128) int8_t smem[];
  __shared__ int s_tile[2];
  const int num_entries = ws[3];
  int* counter = ws;
  int* done = ws + 1;
  const int* prefix = ws + kHdrInts;                 // [num_entries + 1]
  const int* entry = prefix + num_entries + 1;       // [num_entries] = (prob << 2) | cfg

  if (threadIdx.x == 0) s_tile[0] = atomicAdd(counter, 1);
  __syncthreads();
  for (int it = 0;; ++it) {
    const int tile = s_tile[it & 1];
    if (tile >= total_tiles) break;
    if (threadIdx.x == 0) s_tile[(it + 1) & 1] = atomicAdd(counter, 1);   // prefetch next

    // largest j with prefix[j] <= tile (all entries have >= 1 tile)
    int lo = 0, hi = num_entries - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (__ldg(prefix + mid) <= tile) lo = mid; else hi = mid - 1;
    }
    const int e = __ldg(entry + lo);
    const int cfg = e & 3;
    const int local = tile - __ldg(prefix + lo);
    const i8_problem P = probs[e >> 2];
    const int bm = cfg_bm(cfg);
    const int tiles_m = (P.m + bm - 1) / bm;
    const int tm = local % tiles_m, tn = local / tiles_m;   // m-fastest raster (B tile reuse)
    switch (cfg) {
      case 0: run_tile<Cfg0>(smem, P, tm, tn); break;
      case 1: run_tile<Cfg1>(smem, P, tm, tn); break;
      default: run_tile<Cfg2>(smem, P, tm, tn); break;
    }
    __syncthreads();   // makes s_tile[(it+1)&1] visible; run_tile already ended with a barrier
  }
  // Self-reset so the workspace can be relaunched without a memset.
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(done, 1) == (int)gridDim.x - 1) {
      atomicExch(counter, 0);
      atomicExch(done, 0);
      __threadfence();
    }
  }
}

// Second persistent kernel: only tile_cfg 3 (32x32 split-K) tiles, own schedule image.
// A separate __global__ so that i8_grouped_persistent_kernel (cfg0/1/2) keeps exactly the
// round-1 SASS (inlining split-K into the same kernel perturbed cfg0 codegen: job 511001,
// cfg0/auto -3..-5%). Same scheduler logic as the main kernel.
__global__ void __launch_bounds__(kThreads, 2)
    i8_grouped_splitk_kernel(const i8_problem* __restrict__ probs, int* __restrict__ ws,
                             int total_tiles) {
  extern __shared__ __align__(128) int8_t smem[];
  __shared__ int s_tile[2];
  const int num_entries = ws[3];
  int* counter = ws;
  int* done = ws + 1;
  const int* prefix = ws + kHdrInts;
  const int* entry = prefix + num_entries + 1;

  if (threadIdx.x == 0) s_tile[0] = atomicAdd(counter, 1);
  __syncthreads();
  for (int it = 0;; ++it) {
    const int tile = s_tile[it & 1];
    if (tile >= total_tiles) break;
    if (threadIdx.x == 0) s_tile[(it + 1) & 1] = atomicAdd(counter, 1);
    int lo = 0, hi = num_entries - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (__ldg(prefix + mid) <= tile) lo = mid; else hi = mid - 1;
    }
    const int e = __ldg(entry + lo);
    const int local = tile - __ldg(prefix + lo);
    const i8_problem P = probs[e >> 2];
    const int tiles_m = (P.m + Cfg2::BM - 1) / Cfg2::BM;
    run_tile_cfg2_splitk(smem, P, local % tiles_m, local / tiles_m);
    __syncthreads();   // s_tile visibility + reduction buffer free before the next prologue
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(done, 1) == (int)gridDim.x - 1) {
      atomicExch(counter, 0);
      atomicExch(done, 0);
      __threadfence();
    }
  }
}

// ---------------------------------------------------------------- host side
struct DevInfo { int sms = 0; int ctas_per_sm = 0; int ctas_per_sm_sk = 0; bool init = false; };
DevInfo g_dev[64];

const DevInfo& dev_info() {
  int d = 0;
  cudaGetDevice(&d);
  DevInfo& di = g_dev[d & 63];
  if (!di.init) {
    cudaDeviceGetAttribute(&di.sms, cudaDevAttrMultiProcessorCount, d);
    // 48 KB dynamic + a few bytes static: opt in explicitly and ask for the max smem carveout
    // so 2 CTAs/SM fit on sm_120 (~100 KB smem/SM) as well as sm_90.
    cudaFuncSetAttribute(i8_grouped_persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    cudaFuncSetAttribute(i8_grouped_persistent_kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, i8_grouped_persistent_kernel, kThreads,
                                                  kSmemBytes);
    di.ctas_per_sm = occ > 0 ? occ : 1;
    cudaFuncSetAttribute(i8_grouped_splitk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    cudaFuncSetAttribute(i8_grouped_splitk_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);
    occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, i8_grouped_splitk_kernel, kThreads,
                                                  kSmemBytes);
    di.ctas_per_sm_sk = occ > 0 ? occ : 1;
    di.init = true;
  }
  return di;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }
int cfg_tiles(int c, int m, int n) {
  int b = cfg_bm(c);
  return ceil_div(m, b) * ceil_div(n, b);
}
// Relative per-tile cost used only for LPT ordering (k-equivalent overhead 128 per tile for
// prologue/epilogue; uncalibrated placeholder until measured).
double tile_cost(int c, int kp) {
  double b = cfg_bm(c);
  return b * b * (double)(kp + 128);
}

bool valid_problem(const i8_problem& p) {
  if (p.m < 0 || p.n < 0 || p.kp < 0 || (p.kp % 32) != 0 || p.kp > 131040) return false;
  if (p.m == 0 || p.n == 0 || p.kp == 0) return true;   // empty: skipped, C untouched
  if (!p.A || !p.B || !p.C) return false;
  if ((reinterpret_cast<uintptr_t>(p.A) & 15) || (reinterpret_cast<uintptr_t>(p.B) & 15)) return false;
  if (reinterpret_cast<uintptr_t>(p.C) & 3) return false;
  return true;
}

int g_auto_splitk_min_kp = 0x7fffffff;   // auto never picks cfg3 unless enabled

}  // namespace

void i8_set_auto_splitk_min_kp(int kp_min) { g_auto_splitk_min_kp = kp_min; }

// Heuristic: estimated cost = tiles * BM*BN / relative_efficiency(cfg), per problem.
// eff[1] = 0.63 from H200 job 510874 (cfg1/cfg0 TOPS: random64 0.66, ozaki-like 0.62, 8192^3 0.62);
// eff[2] = 0.30 from the same job (0.29). The round-1 per-problem "cannot fill the GPU -> 64x64"
// override was removed (job 510874: auto lost 9% to all-cfg0 on random64); the fill rule is now
// group-level in build_schedule.
int i8_choose_tile_cfg(int m, int n, int kp, int num_sms) {
  (void)kp;
  (void)num_sms;
  if (m <= 0 || n <= 0) return I8_TILE_32x32;
  static const double eff[3] = {1.0, 0.63, 0.30};
  int best = 0;
  double best_cost = 1e300;
  for (int c = 0; c < 3; ++c) {
    double b = cfg_bm(c);
    double cost = cfg_tiles(c, m, n) * b * b / eff[c];
    if (cost < best_cost) { best_cost = cost; best = c; }
  }
  return best;
}

// Workspace = image 1 (main kernel: cfg 0/1/2) followed by image 2 (split-K kernel: cfg 3),
// image 2 starting at an 8-int aligned offset. Each image: 8-int header {counter, done, total,
// ne, ...} + prefix[ne+1] + entry[ne]. Image-1 header also stores {.., [4]=offset2, [5]=total2}.
size_t i8_grouped_gemm_sched_ws_bytes(int max_problems, size_t /*max_tiles*/) {
  if (max_problems < 0) max_problems = 0;
  return sizeof(int) * (size_t)(2 * (size_t)max_problems + 2 * kHdrInts + 2 + 8);
}

namespace {
// Host scratch, sized once by i8_grouped_gemm_reserve() (call it at create time) so the timed
// path never allocates: entry array + merge-sort buffer + staging image for the convenience
// path. Process-wide and mutex-protected (std::mutex lock/unlock does not allocate). If a call
// exceeds the reserved capacity it still works but grows the buffers (reported by
// i8_grouped_gemm_host_allocs()).
struct Ent { double cost; int tiles; int prob; int cfg; };
std::mutex g_scratch_mu;
std::vector<Ent> g_ents, g_tmp;
std::vector<int> g_img;
long long g_host_allocs = 0;

void ensure_capacity(size_t n_probs) {
  if (g_ents.capacity() < n_probs) { g_ents.reserve(n_probs); ++g_host_allocs; }
  if (g_tmp.size() < n_probs) { g_tmp.resize(n_probs); ++g_host_allocs; }
  size_t img_ints = i8_grouped_gemm_sched_ws_bytes((int)n_probs, 0) / sizeof(int);
  if (g_img.size() < img_ints) { g_img.resize(img_ints); ++g_host_allocs; }
}

// LPT order: split-K entries after main-kernel entries; heavier tiles first; more tiles first;
// ties keep input order. Stable bottom-up merge sort on the preallocated g_tmp (no allocation).
bool ent_before(const Ent& a, const Ent& b) {
  bool sa = a.cfg == I8_TILE_32x32_SPLITK, sb = b.cfg == I8_TILE_32x32_SPLITK;
  if (sa != sb) return sb;
  if (a.cost != b.cost) return a.cost > b.cost;
  return a.tiles > b.tiles;
}
void stable_sort_prealloc(Ent* a, Ent* tmp, size_t n) {
  Ent* src = a;
  Ent* dst = tmp;
  for (size_t w = 1; w < n; w *= 2) {
    for (size_t lo = 0; lo < n; lo += 2 * w) {
      size_t mid = std::min(lo + w, n), hi = std::min(lo + 2 * w, n);
      size_t i = lo, j = mid, k = lo;
      while (i < mid && j < hi) dst[k++] = ent_before(src[j], src[i]) ? src[j++] : src[i++];
      while (i < mid) dst[k++] = src[i++];
      while (j < hi) dst[k++] = src[j++];
    }
    std::swap(src, dst);
  }
  if (src != a) std::copy(src, src + n, a);
}
}  // namespace

void i8_grouped_gemm_reserve(int max_problems, size_t /*max_tiles*/) {
  std::lock_guard<std::mutex> lk(g_scratch_mu);
  ensure_capacity(max_problems > 0 ? (size_t)max_problems : 1);
  (void)dev_info();   // also do the one-time cudaFuncSetAttribute / occupancy query here
}
long long i8_grouped_gemm_host_allocs() { return g_host_allocs; }

// Resident CTAs/SM of the kernel that runs tile_cfg (cfg 0/1/2 share the main kernel, so they
// return the same value; cfg 3 = split-K kernel). -1 for an invalid tile_cfg.
int i8_grouped_gemm_ctas_per_sm(int tile_cfg) {
  if (tile_cfg < 0 || tile_cfg >= I8_NUM_TILE_CFGS) return -1;
  const DevInfo& di = dev_info();
  return tile_cfg == I8_TILE_32x32_SPLITK ? di.ctas_per_sm_sk : di.ctas_per_sm;
}

namespace {
// allow_splitk = false (legacy int* API): cfg 3 is mapped to cfg 2, so the schedule only ever
// needs the main kernel (exact round-1 behaviour, 1 launch).
int build_schedule_impl(const i8_problem* h, int num, void* h_ws, size_t ws_bytes,
                        i8_sched_info* info, bool allow_splitk) {
  if (num < 0 || (num > 0 && !h) || !h_ws || !info) return 1;
  if (ws_bytes < i8_grouped_gemm_sched_ws_bytes(num, 0)) return 2;
  std::lock_guard<std::mutex> lk(g_scratch_mu);
  ensure_capacity((size_t)num);
  std::vector<Ent>& ents = g_ents;
  ents.clear();   // keeps capacity
  int sms = 0;
  for (int i = 0; i < num; ++i) {
    if (!valid_problem(h[i])) return 3;
    if (h[i].m == 0 || h[i].n == 0 || h[i].kp == 0) continue;   // skipped (C not written)
    int c = h[i].tile_cfg;
    if (c < 0) {
      if (!sms) sms = dev_info().sms;
      c = i8_choose_tile_cfg(h[i].m, h[i].n, h[i].kp, sms);
      if (c == I8_TILE_32x32 && h[i].kp >= g_auto_splitk_min_kp) c = I8_TILE_32x32_SPLITK;
    }
    if (c >= I8_NUM_TILE_CFGS) return 4;
    if (!allow_splitk && c == I8_TILE_32x32_SPLITK) c = I8_TILE_32x32;
    ents.push_back({tile_cost(c, h[i].kp), cfg_tiles(c, h[i].m, h[i].n), i, c});
  }
  // Group-level fill rule: if the whole group has fewer tiles than SMs, split auto-chosen
  // 128x128 problems into 64x64 tiles (more CTAs busy). Explicit tile_cfg is never changed.
  {
    long long t = 0;
    for (auto& e : ents) t += e.tiles;
    if (!sms) sms = dev_info().sms;
    if (t < sms) {
      for (auto& e : ents) {
        if (e.cfg == 0 && h[e.prob].tile_cfg < 0) {
          e.cfg = 1;
          e.tiles = cfg_tiles(1, h[e.prob].m, h[e.prob].n);
          e.cost = tile_cost(1, h[e.prob].kp);
        }
      }
    }
  }
  // LPT inside each kernel: heaviest tiles first; ties -> more tiles first, then original order.
  // Partition: main-kernel entries (cfg 0/1/2) first, split-K entries (cfg 3) after.
  stable_sort_prealloc(ents.data(), g_tmp.data(), ents.size());
  size_t ne1 = 0;
  while (ne1 < ents.size() && ents[ne1].cfg != I8_TILE_32x32_SPLITK) ++ne1;
  const size_t ne2 = ents.size() - ne1;
  int* w = static_cast<int*>(h_ws);
  auto write_image = [&](int* img, size_t first, size_t ne, long long* tot) {
    int* prefix = img + kHdrInts;
    int* entry = prefix + ne + 1;
    long long t = 0;
    for (size_t j = 0; j < ne; ++j) {
      prefix[j] = (int)t;
      t += ents[first + j].tiles;
      entry[j] = (ents[first + j].prob << 2) | ents[first + j].cfg;
    }
    prefix[ne] = (int)t;
    std::memset(img, 0, sizeof(int) * kHdrInts);
    img[2] = (int)t;
    img[3] = (int)ne;
    *tot = t;
  };
  long long t1 = 0, t2 = 0;
  write_image(w, 0, ne1, &t1);
  const size_t off2 = (kHdrInts + 2 * ne1 + 1 + 7) / 8 * 8;
  if (ne2) write_image(w + off2, ne1, ne2, &t2);
  if (t1 > 0x7fffffffLL - 1000000 || t2 > 0x7fffffffLL - 1000000) return 5;
  w[4] = (int)off2;
  w[5] = (int)t2;
  info->tiles_main = (int)t1;
  info->tiles_splitk = (int)t2;
  info->ws2_offset_ints = (int)off2;
  info->num_launches = (t1 > 0) + (t2 > 0);
  info->bytes_used = sizeof(int) * (ne2 ? off2 + kHdrInts + 2 * ne2 + 1 : kHdrInts + 2 * ne1 + 1);
  return 0;
}
}  // namespace

int i8_grouped_gemm_build_schedule(const i8_problem* h, int num, void* h_ws, size_t ws_bytes,
                                   i8_sched_info* info) {
  return build_schedule_impl(h, num, h_ws, ws_bytes, info, true);
}

int i8_grouped_gemm_build_schedule(const i8_problem* h, int num, void* h_ws, size_t ws_bytes,
                                   int* total_tiles) {
  if (!total_tiles) return 1;
  i8_sched_info info;
  int rc = build_schedule_impl(h, num, h_ws, ws_bytes, &info, false);
  if (rc) return rc;
  *total_tiles = info.tiles_main;
  return 0;
}

int i8_grouped_gemm_launch_count(const i8_sched_info* info) {
  return info ? info->num_launches : 0;
}

int i8_grouped_gemm_launch_count(const i8_problem* h, int num) {
  bool main_k = false, sk = false;
  for (int i = 0; i < num; ++i) {
    if (h[i].m <= 0 || h[i].n <= 0 || h[i].kp <= 0) continue;
    int c = h[i].tile_cfg;
    if (c < 0) {
      c = i8_choose_tile_cfg(h[i].m, h[i].n, h[i].kp, 0);
      if (c == I8_TILE_32x32 && h[i].kp >= g_auto_splitk_min_kp) c = I8_TILE_32x32_SPLITK;
    }
    if (c == I8_TILE_32x32_SPLITK) sk = true; else main_k = true;
  }
  return (int)main_k + (int)sk;
}

int i8_grouped_gemm_launch(const i8_problem* d_probs, int num, const void* d_sched_ws,
                           const i8_sched_info* info, cudaStream_t s) {
  (void)num;
  if (!info) return 1;
  if (info->num_launches == 0) return 0;
  if (!d_probs || !d_sched_ws) return 1;
  const DevInfo& di = dev_info();
  int* ws = const_cast<int*>(static_cast<const int*>(d_sched_ws));
  (void)cudaGetLastError();   // clear a stale (non-sticky) error so we only report our launches
  // Split-K (cfg3, light small-problem tiles) first: its tail is short; the heavy main kernel
  // follows on the same stream. Outputs and workspace images are disjoint.
  if (info->tiles_splitk > 0) {
    long long grid = (long long)di.sms * di.ctas_per_sm_sk;
    if (grid > info->tiles_splitk) grid = info->tiles_splitk;
    i8_grouped_splitk_kernel<<<(int)grid, kThreads, kSmemBytes, s>>>(
        d_probs, ws + info->ws2_offset_ints, info->tiles_splitk);
  }
  if (info->tiles_main > 0) {
    long long grid = (long long)di.sms * di.ctas_per_sm;
    if (grid > info->tiles_main) grid = info->tiles_main;
    i8_grouped_persistent_kernel<<<(int)grid, kThreads, kSmemBytes, s>>>(d_probs, ws,
                                                                          info->tiles_main);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 6;
}

int i8_grouped_gemm_launch(const i8_problem* d_probs, int num, const void* d_sched_ws,
                           int total_tiles, cudaStream_t s) {
  i8_sched_info info{};
  info.tiles_main = total_tiles > 0 ? total_tiles : 0;
  info.num_launches = info.tiles_main > 0;
  return i8_grouped_gemm_launch(d_probs, num, d_sched_ws, &info, s);
}

int i8_grouped_gemm_ex(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                       void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s, int* num_launches) {
  if (num_launches) *num_launches = 0;
  if (num == 0) return 0;
  if (!d_sched_ws) return 1;
  // Host staging image. Pageable cudaMemcpyAsync returns only after the source has been staged,
  // so reusing this buffer on the next call is safe. Host-side growth only (never device malloc);
  // callers wanting a fully async copy use build_schedule into their own pinned buffer.
  size_t need = i8_grouped_gemm_sched_ws_bytes(num, 0);
  if (need > sched_ws_bytes) return 2;
  std::vector<int>& img = g_img;   // reserved by i8_grouped_gemm_reserve
  {
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    ensure_capacity((size_t)num);
  }
  // NOTE: g_img is shared; concurrent convenience calls from several host threads are not
  // supported (use build_schedule into caller-owned buffers for that).
  i8_sched_info info;
  int rc = i8_grouped_gemm_build_schedule(h_probs_for_sched, num, img.data(), need, &info);
  if (rc) return rc;
  if (info.num_launches == 0) return 0;
  if (cudaMemcpyAsync(d_sched_ws, img.data(), info.bytes_used, cudaMemcpyHostToDevice, s) !=
      cudaSuccess)
    return 7;
  rc = i8_grouped_gemm_launch(d_probs, num, d_sched_ws, &info, s);
  if (!rc && num_launches) *num_launches = info.num_launches;
  return rc;
}

int i8_grouped_gemm(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                    void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s) {
  return i8_grouped_gemm_ex(d_probs, num, h_probs_for_sched, d_sched_ws, sched_ws_bytes, s,
                            nullptr);
}

// ---------------------------------------------------------------- cuBLASLt reference
// Our layout is exactly cuBLASLt's int8 "TN": A stored col-major kp x m (lda = kp) with
// op(A) = T, B col-major kp x n (ldb = kp) with op(B) = N, C col-major m x n (ldc = m).
// Descriptors are created per call (reference/baseline only; host-side allocations).
int i8_gemm_cublaslt_single(void* lt_handle, int m, int n, int kp, const int8_t* A,
                            const int8_t* B, int32_t* C, void* d_workspace, size_t ws_bytes,
                            cudaStream_t s) {
  cublasLtHandle_t lt = static_cast<cublasLtHandle_t>(lt_handle);
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasStatus_t st;
  int rc = 0;
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  const int32_t alpha = 1, beta = 0;
  cublasLtMatmulHeuristicResult_t heur;
  int nres = 0;
#define LT_CHECK(x) do { st = (x); if (st != CUBLAS_STATUS_SUCCESS) { rc = (int)st; goto done; } } while (0)
  LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, kp, m, kp));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, kp, n, kp));
  LT_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, m, n, m));
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ws_bytes, sizeof(ws_bytes)));
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &nres));
  if (nres == 0) { rc = (int)CUBLAS_STATUS_NOT_SUPPORTED; goto done; }
  LT_CHECK(cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc, &heur.algo,
                          d_workspace, ws_bytes, s));
#undef LT_CHECK
done:
  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (op) cublasLtMatmulDescDestroy(op);
  return rc;
}
