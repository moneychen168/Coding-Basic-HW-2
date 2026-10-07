// ozaki2_numerics.cu -- Ozaki scheme II (CRT, INT8 residues), numerical part.  Owner: agent-N.
//
// Reference: K. Ozaki, Y. Uchino, T. Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of
// floating-point matrix multiplication using an integer modular technique", arXiv:2504.08009.
// Written from the paper; no code taken from GEMMul8.
//
// ----------------------------------------------------------------------------------------------
// Method (per problem, N moduli p_0..p_{N-1}, P = prod p_l; notation as in moduli.h)
//
// 1. Scaling ("fast mode" of the paper, Cauchy-Schwarz bound).
//    s_i = row exponent of A, t_j = column exponent of B, chosen maximal such that
//        2^{s_i} ||A(i,:)||_2 <= T   and   2^{t_j} ||B(:,j)||_2 <= T,
//        T = sqrt(P/2 (1-2^-20)) (1-2^-40).
//    A'_{ik} = trunc(2^{s_i} A_{ik}),  B'_{kj} = trunc(2^{t_j} B_{kj})   (round toward zero),
//    so |A'| <= 2^{s} |A| elementwise and, by Cauchy-Schwarz,
//        sum_k |A'_{ik}| |B'_{kj}| <= ||A'(i,:)|| ||B'(:,j)|| <= T^2 < P/2 (1-2^-20).        (*)
//    (*) is exactly the paper's no-overflow condition 2 c_max < P, with margin 2^-20 that
//    (a) absorbs the rounding error of the computed 2-norms (relative <= k u < 2^-36) and
//    (b) keeps |X/P| away from 1/2 so the rounding of Q in step 4 is unambiguous.
//    The paper's simpler bound k * max|A'| * max|B'| < P/2 implies (*) (||a||_2 <= sqrt(k) max|a|),
//    so the Cauchy-Schwarz choice never truncates more bits than the k-based one.
//    Because s_i is maximal, 2^{s_i} ||A(i,:)|| > T/2 (same for B).
//
// 2. Residues.  |A'_{ik}| < 2^81, so A' = sign * M * 2^E with M < 2^53 an integer, E >= 0.
//    M = c0 + c1 2^18 + c2 2^36 (18-bit chunks) and
//        A' mod p = sign * (c0 [2^E]_p + c1 [2^{E+18}]_p + c2 [2^{E+36}]_p) mod p,
//    with [2^e]_p tabulated (moduli.h).  The sum is < 2^28, reduced by a 32-bit Barrett step.
//    The residue r in [0,p) is mapped to the symmetric int8 range: r > (p-1)/2 -> r - p
//    (gives [-128,127] for p = 256, [-(p-1)/2,(p-1)/2] for odd p).
//
// 3. (agent-K) C_l = A'_l B'_l exactly in INT32 (needs kp * 128^2 < 2^31; we require k <= 131040,
//    i.e. kp <= 131040 < 2^17).
//
// 4. CRT reconstruction.  y_l = (C_l mod p_l) * q_l mod p_l in [0,p_l),
//        S = sum_l M_l y_l  (0 <= S < N P),   X = S - Q P with Q = round(S/P),
//    and by (*) X = (A'B')_{ij} exactly because |A'B'| < P/2.
//    Evaluation without wide integers or FP64-heavy double-double loops: with u1 = 2^{sh_N},
//        M_l / u1 = s1_l + s2_l 2^-18 + d_l,  |d_l| <= 2^-19,  s1_l < 2^49 (uint64), s2_l <= 2^18,
//        S1 = sum s1_l y_l  (uint64, exact, < 20*255*2^49 < 2^63),
//        S2 = sum s2_l y_l  (uint32, exact, <= 20*255*2^18 < 2^31),
//        Q  = rint((S1 + S2 2^-18) * u1/P)                      (exact integer, see margin above)
//        X/u1 = (S1 - Q P1) + (S2 2^-18 - Q P2)                  (first bracket exact in int64)
//    This is a (64-bit integer + double) representation of X, i.e. a double-double-class
//    evaluation with absolute error |dX| <= u1 (N 255 2^-19 + 2^-10) + 2u|X|.
//    For N <= 6, u1 < 1 and everything up to the final conversion is exact.
//
// 5. C_ij = alpha * 2^{-(s_i+t_j)} X + beta * C_ij   (power-of-two scaling is exact).
//
// Error bound (used by oz2_choose_num_moduli), relative to ||A(i,:)||_2 ||B(:,j)||_2:
//    With dA = 2^s A - A' (|dA| < 1 elementwise, ||dA(i,:)|| < sqrt(k)), same for B,
//      |2^{s+t} (AB)_{ij} - (A'B')_{ij}| <= ||dA|| ||B'|| + ||A'|| ||dB|| + ||dA|| ||dB||,
//    with ||B'|| <= 2^t ||b||, ||A'|| <= 2^s ||a||, ||dA||,||dB|| < sqrt(k), 2^s||a||, 2^t||b|| > T/2:
//      term1 / (2^{s+t}||a|| ||b||) <= sqrt(k) / (2^s ||a||) < 2 sqrt(k)/T   (term2 likewise),
//      term3 / (2^{s+t}||a|| ||b||) < k / (T^2/4).
//    Hence  |C_ij - (AB)_ij| / (||A(i,:)|| ||B(:,j)||)
//          <= eps_N(k) = 4 sqrt(k)/T + 4k/T^2 + u   [+ 4 u1 (N 255 2^-19 + 2^-10)/T^2, the CRT
//             error of step 4, ~1e-19, kept for rigor],   u = 2^-53 (final rounding).
//    Since T ~ sqrt(P/2), the first term is ~ 4 sqrt(2k/P): each extra modulus (~7.6-8 bits)
//    gains ~4 bits.  For target 1e-15 this gives N = 15 for k = 64..4096, N = 16 for k >= 8192.
//    Targets below ~u are unreachable (returns -1).
//
// Non-finite input (NaN/Inf) in A(i,:) or B(:,j): the row/column gets shift
// OZ2_SHIFT_NONFINITE, its residues are 0, the affected C entries are written as NaN, and the
// sticky device counter read by oz2_nonfinite_count() is incremented so the caller can rerun
// that problem with native FP64.  All-zero rows/columns are exact (shift 0, residues 0).  (Bound is normwise per entry; it is >= the componentwise error relative to
//    (|A||B|)_ij scaled by ||a|| ||b|| / (|A||B|)_ij, which is what the tests report.)
// ----------------------------------------------------------------------------------------------
#include "ozaki2_numerics.h"
#include "moduli.h"

#include <cmath>
#include <cstdio>

namespace {

__constant__ oz2_crt_tables c_tab;
__device__ int g_nonfinite;  // sticky count of non-finite rows/columns seen by prep

const oz2_crt_tables& host_tables(int* status = nullptr) {
  static int st = -1;
  static oz2_crt_tables tab = [] { oz2_crt_tables t; st = oz2_build_crt_tables(&t); return t; }();
  if (status) *status = st;
  return tab;
}

constexpr int A_ROWS = 32;    // rows of A per block
constexpr int A_KT = 32;      // k tile for the A transpose
constexpr int A_PITCH = 36;   // bytes per row in the smem tile (odd number of words: no conflicts)
constexpr int B_COLS = 8;     // columns of B per block (one warp each)
constexpr int THREADS = 256;

// x * 2^e, exact unless the result is subnormal; |e| <= 3000.
__device__ __forceinline__ double mul_pow2(double x, int e) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    int ee = max(-1000, min(1000, e));
    x *= __longlong_as_double((long long)(ee + 1023) << 52);
    e -= ee;
  }
  return x;
}

__device__ __forceinline__ uint32_t mod_barrett(uint32_t u, uint32_t p, uint32_t magic) {
  uint32_t q = __umulhi(u, magic);
  uint32_t r = u - q * p;  // r < 2p (see header of moduli.h: magic = floor(2^32/p))
  r = r >= p ? r - p : r;
  r = r >= p ? r - p : r;
  return r;
}

// Largest s with 2^s * nub <= T, where nub = f * 2^g (f in [0.5,1)), T = F * 2^G.
__device__ __forceinline__ int choose_shift(double nrm_mant, int nrm_exp_add, double T) {
  int g, G;
  double f = frexp(nrm_mant, &g);
  double F = frexp(T, &G);
  g += nrm_exp_add;
  return G - g - (f > F ? 1 : 0);
}

// Row/col shift from (maxabs, sum of squares, rescaled sum of squares if needed).
// ssq is either the plain sum of squares (rescaled == false) or sum (a 2^-e)^2 with
// e = ilogb(maxabs) (rescaled == true).
__device__ __forceinline__ int shift_from_norm(double maxabs, double ssq, bool rescaled, double T) {
  if (isinf(maxabs) || isnan(ssq)) {  // Inf (maxabs) or NaN (propagates into ssq)
    atomicAdd(&g_nonfinite, 1);
    return OZ2_SHIFT_NONFINITE;
  }
  if (!(maxabs > 0.0)) return 0;  // zero row/column (or NaN): all residues 0
  const double margin = 1.0 + 9.313225746154785e-10;  // 1 + 2^-30
  double nm = sqrt(ssq * margin);
  int add = rescaled ? ilogb(maxabs) : 0;
  return choose_shift(nm, add, T);
}

__device__ __forceinline__ bool needs_rescale(double maxabs) {
  // plain squares are safe (no overflow, harmless underflow) iff maxabs in [2^-400, 2^400]
  return maxabs > 0.0 && (maxabs > 0x1p400 || maxabs < 0x1p-400);
}

struct Dec {
  uint32_t c0, c1, c2;
  int E;
  bool neg;
};

__device__ __forceinline__ Dec decompose(double x) {  // x integer valued
  Dec d;
  double ax = fabs(x);
  d.neg = x < 0.0;
  uint64_t M;
  int E;
  if (ax < 9007199254740992.0) {  // 2^53
    M = (uint64_t)ax;
    E = 0;
  } else {
    uint64_t b = (uint64_t)__double_as_longlong(ax);
    E = (int)((b >> 52) & 0x7ff) - 1023 - 52;
    M = (b & ((1ull << 52) - 1)) | (1ull << 52);
    E = min(E, OZ2_POW2_TABLE - 37);  // guaranteed E <= 28 by |x| <= T < 2^81; clamp for safety
  }
  d.c0 = (uint32_t)(M & 0x3ffff);
  d.c1 = (uint32_t)((M >> 18) & 0x3ffff);
  d.c2 = (uint32_t)(M >> 36);
  d.E = E;
  return d;
}

__device__ __forceinline__ int8_t residue_sym(const Dec& d, int l, const uint8_t (*pw)[OZ2_POW2_TABLE]) {
  const uint32_t p = (uint32_t)c_tab.p[l];
  uint32_t t = d.c0 * pw[l][d.E] + d.c1 * pw[l][d.E + 18] + d.c2 * pw[l][d.E + 36];  // < 2^28
  uint32_t r = mod_barrett(t, p, c_tab.magic[l]);
  if (d.neg && r) r = p - r;
  int ri = (int)r;
  if (ri > (int)((p - 1) >> 1)) ri -= (int)p;
  return (int8_t)ri;
}

// trunc (round toward zero), NOT rint: guarantees |A'| <= 2^s |A| so ||A'(i,:)|| <= T.
__device__ __forceinline__ double scaled_trunc(double a, int s) {
  return s == OZ2_SHIFT_NONFINITE ? 0.0 : trunc(mul_pow2(a, s));
}

// ------------------------------------------------------------------------------- prep kernel
__global__ void __launch_bounds__(THREADS) oz2_prep_kernel(const oz2_problem* __restrict__ probs, int a_blocks) {
  __shared__ uint8_t s_pow[GGEMM_MAX_MODULI][OZ2_POW2_TABLE];
  __shared__ __align__(16) int8_t s_tile[GGEMM_MAX_MODULI][A_ROWS][A_PITCH];
  __shared__ double s_max[THREADS / 32][32];
  __shared__ double s_ssq[THREADS / 32][32];
  __shared__ int s_shift[A_ROWS];

  const int tid = threadIdx.x;
  for (int i = tid; i < GGEMM_MAX_MODULI * OZ2_POW2_TABLE; i += THREADS)
    (&s_pow[0][0])[i] = c_tab.pow2[i / OZ2_POW2_TABLE][i % OZ2_POW2_TABLE];
  __syncthreads();

  const oz2_problem& pr = probs[blockIdx.y];
  const int N = pr.num_moduli;
  const int K = pr.k, Kp = pr.kp;
  if (N < 1 || N > GGEMM_MAX_MODULI) return;
  const double T = c_tab.T[N];

  if ((int)blockIdx.x < a_blocks) {
    // ------------------------------------------------ A: 32 rows, transpose to row-major
    const int row0 = blockIdx.x * A_ROWS;
    const int m = pr.m;
    if (row0 >= m) return;
    const int tx = tid & 31, ty = tid >> 5;
    const int i = row0 + tx;
    const bool valid = i < m;
    const double* __restrict__ A = pr.A;
    const size_t lda = (size_t)pr.lda;

    double mx = 0.0, ss = 0.0;
    if (valid)
      for (int k = ty; k < K; k += THREADS / 32) {
        double a = A[i + k * lda];
        mx = fmax(mx, fabs(a));
        ss = fma(a, a, ss);
      }
    s_max[ty][tx] = mx;
    s_ssq[ty][tx] = ss;
    __syncthreads();
    if (ty == 0) {
      double M = 0.0;
      for (int r = 0; r < THREADS / 32; ++r) M = fmax(M, s_max[r][tx]);
      s_max[0][tx] = M;
    }
    __syncthreads();
    const double rowmax = s_max[0][tx];
    const bool resc = valid && needs_rescale(rowmax);
    if (resc) {  // rare path: second pass with exact power-of-two prescaling
      const int e = -ilogb(rowmax);
      ss = 0.0;
      for (int k = ty; k < K; k += THREADS / 32) {
        double a = mul_pow2(A[i + k * lda], e);
        ss = fma(a, a, ss);
      }
    }
    __syncthreads();
    s_ssq[ty][tx] = ss;
    __syncthreads();
    if (ty == 0) {
      double S = 0.0;
      for (int r = 0; r < THREADS / 32; ++r) S += s_ssq[r][tx];
      int sh = valid ? shift_from_norm(rowmax, S, resc, T) : 0;
      s_shift[tx] = sh;
      if (valid) pr.row_shift[i] = sh;
    }
    __syncthreads();
    const int sh = s_shift[tx];

    const int orow = tid >> 3, ow = tid & 7;  // write-out mapping: 32 rows x 8 words
    const bool ovalid = row0 + orow < m;
    for (int k0 = 0; k0 < Kp; k0 += A_KT) {
#pragma unroll
      for (int r = 0; r < A_KT / (THREADS / 32); ++r) {
        const int kk = ty + r * (THREADS / 32);
        const int k = k0 + kk;
        double a = (valid && k < K) ? A[i + k * lda] : 0.0;
        Dec d = decompose(scaled_trunc(a, sh));
        for (int l = 0; l < N; ++l) s_tile[l][tx][kk] = residue_sym(d, l, s_pow);
      }
      __syncthreads();
      if (ovalid) {
        int8_t* dst = pr.A8 + (size_t)(row0 + orow) * Kp + k0 + ow * 4;
        for (int l = 0; l < N; ++l)
          *reinterpret_cast<int32_t*>(dst + l * pr.a8_mod_stride) =
              *reinterpret_cast<const int32_t*>(&s_tile[l][orow][ow * 4]);
      }
      __syncthreads();
    }
  } else {
    // ------------------------------------------------ B: one warp per column (already K-contiguous)
    const int lane = tid & 31, w = tid >> 5;
    const int j = (blockIdx.x - a_blocks) * B_COLS + w;
    if (j >= pr.n) return;  // no __syncthreads below
    const double* __restrict__ b = pr.B + (size_t)j * pr.ldb;

    double mx = 0.0, ss = 0.0;
    for (int k = lane; k < K; k += 32) {
      double a = b[k];
      mx = fmax(mx, fabs(a));
      ss = fma(a, a, ss);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const bool resc = needs_rescale(mx);
    if (resc) {
      const int e = -ilogb(mx);
      ss = 0.0;
      for (int k = lane; k < K; k += 32) {
        double a = mul_pow2(b[k], e);
        ss = fma(a, a, ss);
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);  // butterfly: identical in all lanes
    const int sh = shift_from_norm(mx, ss, resc, T);
    if (lane == 0) pr.col_shift[j] = sh;

    int8_t* base = pr.B8 + (size_t)j * Kp;
    for (int k0 = 4 * lane; k0 < Kp; k0 += 128) {
      Dec d[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int k = k0 + e;
        d[e] = decompose(scaled_trunc(k < K ? b[k] : 0.0, sh));
      }
      for (int l = 0; l < N; ++l) {
        uint32_t packed = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) packed |= (uint32_t)(uint8_t)residue_sym(d[e], l, s_pow) << (8 * e);
        *reinterpret_cast<uint32_t*>(base + l * pr.b8_mod_stride + k0) = packed;
      }
    }
  }
}

// ------------------------------------------------------------------------ reconstruct kernel
__global__ void __launch_bounds__(THREADS) oz2_reconstruct_kernel(const oz2_problem* __restrict__ probs) {
  const oz2_problem& pr = probs[blockIdx.y];
  const int N = pr.num_moduli;
  if (N < 1 || N > GGEMM_MAX_MODULI) return;
  const int m = pr.m;
  const size_t total = (size_t)m * (size_t)pr.n;
  const int32_t* __restrict__ C32 = pr.C32;
  const size_t cst = pr.c32_mod_stride;
  const double alpha = pr.alpha, beta = pr.beta;
  const int64_t P1 = c_tab.P1[N];
  const double P2 = c_tab.P2[N], invPu = c_tab.inv_Pu[N];
  const int lu1 = c_tab.log2_u1[N];

  for (size_t idx = (size_t)blockIdx.x * THREADS + threadIdx.x; idx < total; idx += (size_t)gridDim.x * THREADS) {
    uint64_t S1 = 0;
    uint32_t S2 = 0;
    for (int l = 0; l < N; ++l) {
      const uint32_t p = (uint32_t)c_tab.p[l], mg = c_tab.magic[l];
      const int32_t c = C32[l * cst + idx];
      const uint32_t u = c < 0 ? 0u - (uint32_t)c : (uint32_t)c;
      uint32_t r = mod_barrett(u, p, mg);
      if (c < 0 && r) r = p - r;
      const uint32_t y = mod_barrett(r * (uint32_t)c_tab.qinv[N][l], p, mg);
      S1 += c_tab.s1[N][l] * (uint64_t)y;
      S2 += c_tab.s2[N][l] * y;
    }
    const double s2d = (double)S2 * (1.0 / (double)(1u << OZ2_S2_BITS));
    const double Q = rint(((double)S1 + s2d) * invPu);
    const int64_t X1 = (int64_t)S1 - (int64_t)Q * P1;
    const double xr = (double)X1 + (s2d - Q * P2);  // X / u1
    const int i = (int)(idx % (size_t)m);
    const int j = (int)(idx / (size_t)m);
    const int rsh = pr.row_shift[i], csh = pr.col_shift[j];
    double* cp = pr.C + i + (size_t)j * pr.ldc;
    if (rsh == OZ2_SHIFT_NONFINITE || csh == OZ2_SHIFT_NONFINITE) { *cp = __longlong_as_double(0x7ff8000000000000ll); continue; }
    const double v = mul_pow2(xr, lu1 - rsh - csh);
    *cp = (beta == 0.0) ? alpha * v : fma(alpha, v, beta * *cp);
  }
}

int g_uploaded_dev = -1;

}  // namespace

// ------------------------------------------------------------------------------------- host
int oz2_init_constants() {
  int st;
  const oz2_crt_tables& t = host_tables(&st);
  if (st != 0) return 100 + st;
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess) return 1;
  if (dev == g_uploaded_dev) return 0;
  if (cudaMemcpyToSymbol(c_tab, &t, sizeof(t)) != cudaSuccess) return 2;
  g_uploaded_dev = dev;
  return 0;
}

int oz2_prep_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s) {
  if (num <= 0) return 0;
  if (num > 65535 || max_m < 0 || max_n < 0) return 1;
  if (int e = oz2_init_constants()) return e;
  const int a_blocks = (max_m + A_ROWS - 1) / A_ROWS;
  const int b_blocks = (max_n + B_COLS - 1) / B_COLS;
  if (a_blocks + b_blocks == 0) return 0;
  oz2_prep_kernel<<<dim3(a_blocks + b_blocks, num), THREADS, 0, s>>>(d, a_blocks);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int oz2_reconstruct_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s) {
  if (num <= 0) return 0;
  if (num > 65535 || max_m < 0 || max_n < 0) return 1;
  if (int e = oz2_init_constants()) return e;
  const size_t mn = (size_t)max_m * (size_t)max_n;
  if (mn == 0) return 0;
  size_t gx = (mn + THREADS * 4 - 1) / (THREADS * 4);  // ~4 elements per thread
  if (gx > 1024) gx = 1024;
  oz2_reconstruct_kernel<<<dim3((unsigned)gx, num), THREADS, 0, s>>>(d);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

double oz2_error_bound(int k, int num_moduli) {
  int st;
  const oz2_crt_tables& t = host_tables(&st);
  if (st != 0 || num_moduli < 1 || num_moduli > GGEMM_MAX_MODULI || k < 0) return INFINITY;
  const double T = t.T[num_moduli];
  const double T2 = T * T;
  const double kk = (double)k;
  const double u = std::ldexp(1.0, -53);
  const double u1 = std::ldexp(1.0, t.log2_u1[num_moduli]);
  return 4.0 * std::sqrt(kk) / T + 4.0 * kk / T2 +
         4.0 * u1 * (num_moduli * 255.0 * std::ldexp(1.0, -19) + std::ldexp(1.0, -10)) / T2 + u;
}

int oz2_choose_num_moduli(int k, double target_rel_err) {
  if (k < 0 || k > OZ2_MAX_K) return -1;  // INT32 accumulation: kp * 128^2 must stay < 2^31
  for (int N = 2; N <= GGEMM_MAX_MODULI; ++N)
    if (oz2_error_bound(k, N) <= target_rel_err) return N;
  return -1;
}

int oz2_nonfinite_count(int* count, int reset, cudaStream_t s) {
  if (!count) return 1;
  if (cudaMemcpyFromSymbolAsync(count, g_nonfinite, sizeof(int), 0, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  if (cudaStreamSynchronize(s) != cudaSuccess) return 3;
  if (reset && *count) {
    const int zero = 0;
    if (cudaMemcpyToSymbol(g_nonfinite, &zero, sizeof(int)) != cudaSuccess) return 4;
  }
  return 0;
}
