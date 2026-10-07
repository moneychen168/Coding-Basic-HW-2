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
//    The norm is computed from plain squares when the row max is in [2^-400, 2^400] (no
//    overflow possible: k * 2^800 < 2^1024), otherwise from squares of (a / 2^ilogb(rowmax)).
//    A'_{ik} = trunc(2^{s_i} A_{ik}),  B'_{kj} = trunc(2^{t_j} B_{kj})   (round toward zero,
//    never rint: rint could make ||A'(i,:)|| exceed T), so |A'| <= 2^{s} |A| and
//        sum_k |A'_{ik}| |B'_{kj}| <= ||A'(i,:)|| ||B'(:,j)|| <= T^2 < P/2 (1-2^-20).        (*)
//    (*) is the paper's no-overflow condition 2 c_max < P, with margin 2^-20 that
//    (a) absorbs the rounding error of the computed 2-norms (relative <= k u < 2^-36) and
//    (b) keeps |X/P| away from 1/2 so the rounding of Q in step 4 is unambiguous.
//    The paper's simpler bound k * max|A'| * max|B'| < P/2 implies (*), so the Cauchy-Schwarz
//    choice never truncates more bits.  Because s_i is maximal, 2^{s_i} ||A(i,:)|| > T/2.
//
// 2. Residues.  |A'_{ik}| < 2^81, so A' = sign * M * 2^E with M < 2^53 an integer, E >= 0.
//    M = c0 + c1 2^18 + c2 2^36 (18-bit chunks) and
//        A' mod p = sign * (c0 [2^E]_p + c1 [2^{E+18}]_p + c2 [2^{E+36}]_p) mod p,
//    with [2^e]_p tabulated (moduli.h).  The sum is < 2^28, reduced by a 32-bit Barrett step
//    (magic = floor(2^32/p) underestimates the quotient by at most 1 -> conditional subtract).
//    The residue r in [0,p) is mapped to the symmetric int8 range: r > (p-1)/2 -> r - p
//    (p = 256: 128..255 -> -128..-1; odd p: [-(p-1)/2,(p-1)/2]).
//
// 3. (agent-K) C_l = A'_l B'_l exactly in INT32 (needs kp * 128^2 < 2^31; we require
//    k <= OZ2_MAX_K = 131040).
//
// 4. CRT reconstruction.  y_l = (C_l mod p_l) * q_l mod p_l in [0,p_l) (C_l < 0 handled by
//    reducing |C_l| and negating), S = sum_l M_l y_l, X = S - Q P with Q = round(S/P);
//    by (*) X = (A'B')_{ij} exactly.  With u1 = 2^{sh_N},
//        M_l / u1 = s1_l + s2_l 2^-18 + d_l,  |d_l| <= 2^-19,  s1_l < 2^49 (uint64), s2_l <= 2^18,
//        S1 = sum s1_l y_l  (uint64, exact, < 2^63),  S2 = sum s2_l y_l  (uint32, exact, < 2^31),
//        Q  = rint((S1 + S2 2^-18) * u1/P),   X/u1 = (S1 - Q P1) + (S2 2^-18 - Q P2)
//    (an int64 + double = double-double-class evaluation), |dX| <= u1 (N 255 2^-19 + 2^-10) + 2u|X|.
//
// 5. C_ij = alpha * 2^{-(s_i+t_j)} X + beta * C_ij   (power-of-two scaling is exact).
//
// Normwise error bound (oz2_error_bound / oz2_choose_num_moduli), relative to ||a_i|| ||b_j||:
//    with dA = 2^s A - A' (|dA| < 1, ||dA(i,:)|| < sqrt(k)), same for B,
//      |2^{s+t}(AB)_ij - (A'B')_ij| <= ||dA|| ||B'|| + ||A'|| ||dB|| + ||dA|| ||dB||,
//    ||B'|| <= 2^t||b||, 2^s||a|| > T/2  =>  term1/(2^{s+t}||a|| ||b||) < 2 sqrt(k)/T, same term2,
//    term3 < 4k/T^2:   eps_N(k) = 4 sqrt(k)/T + 4k/T^2 + u  (+ CRT term ~1e-19).
//    Each extra modulus (~7.6-8 bits) gains ~4 bits; target 1e-15 -> N = 15 (k = 64..4096).
//
// ---- Componentwise contract: certify-and-repair (step 6) ------------------------------------
// The gate is the classical FP64 bound  |C_ij - Cref_ij| <= gamma_n (|alpha|(|A||B|)_ij + |beta||C0_ij|),
// n = k (+1 if alpha != 1) (+1 if beta != 0), gamma_n = n u/(1 - n u).  The normwise bound above
// does not imply it: rho_ij = ||a_i|| ||b_j|| / (|A||B|)_ij can be arbitrarily large (heavy tailed
// or disjoint-support rows/columns), and no fixed N can cover every rho.  Hence, per entry:
//  (i)  a rigorous error bound  E_ij = 2^{-(s+t)} [ da_i nb_j + na_i db_j + da_i db_j
//                                     + u1 (N 255 2^-19 + 2^-10) + 4u|X| ]
//       with da_i >= ||dA(i,:)||_2, na_i >= ||A'(i,:)||_2 (float, rounded up, computed in prep);
//  (ii) a rigorous LOWER bound of (|A||B|)_ij from one extra INT8 plane ("abs plane", index N):
//       A''_ik = floor(2^{sig_i}|A_ik|) in [0,127] (sig_i = 6 - ilogb(rowmax)), same for B,
//       L_ij = (A''B'')_ij exact in INT32 (k*127^2 < 2^31)  =>  (|A||B|)_ij >= 2^{-(sig_i+tau_j)} L_ij;
//       also (|A||B|)_ij >= |AB|_ij >= |v_ij| - E_ij.
//  (iii) if  |alpha| E_ij (1+2u) <= (gamma_n - 2u) (|alpha| Lab_ij + |beta C0_ij|)  (with 2^-40
//       safety factors) the emulated value is certified to meet the gate; otherwise the entry is
//       "repaired": recomputed in the same thread as a sequential FP64 FMA dot product, which
//       satisfies the same gamma_n bound as native DGEMM (Higham, Sec. 3.5).
//  Rows/columns with NaN/Inf are always repaired (IEEE propagation identical to a native dot),
//  all-zero rows/columns give C = beta C0 exactly.  Repairs are counted (oz2_read_counters);
//  the planner chooses N so that repairs are rare (oz2_choose_num_moduli_cw).
// ----------------------------------------------------------------------------------------------
#include "ozaki2_numerics.h"
#include "moduli.h"

#include <cmath>
#include <cstdio>

namespace {

__constant__ oz2_crt_tables c_tab;
// Sticky device counters (reset on a stream by oz2_read_counters):
//   [0] rows/columns containing NaN/Inf (prep)          [1] uncertified entries (COUNT mode)
//   [2] entries repaired in FP64 (ON mode)               [3] safety-fallback entries (NaN/Inf rows/cols)
__device__ unsigned long long g_cnt[4];

const oz2_crt_tables& host_tables(int* status = nullptr) {
  static int st = -1;
  static oz2_crt_tables tab = [] { oz2_crt_tables t; st = oz2_build_crt_tables(&t); return t; }();
  if (status) *status = st;
  return tab;
}

constexpr int A_ROWS = 32;    // rows of A per block
constexpr int A_KT = 32;      // k tile for the A transpose
// smem tile: 32 rows x 32 bytes (8 words) per plane, word index XOR-swizzled by (row >> 2) & 7
// so that both the byte stores (fixed k, 32 rows) and the word loads (4 rows x 8 words per warp)
// hit 32 distinct banks.
__device__ __forceinline__ int tile_word(int row, int w) { return row * 8 + (w ^ ((row >> 2) & 7)); }
constexpr int B_COLS = 8;     // columns of B per block (one warp each)
constexpr int THREADS = 256;
constexpr int NPLANES = GGEMM_MAX_MODULI + OZ2_EXTRA_PLANES;

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
  uint32_t r = u - q * p;  // magic = floor(2^32/p): q >= true quotient - 1, so r < 2p
  r = r >= p ? r - p : r;
  r = r >= p ? r - p : r;  // defensive second step
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

__device__ __forceinline__ bool row_nonfinite(double maxabs, double ssq) {
  return isinf(maxabs) || isnan(ssq);  // Inf shows in maxabs, NaN propagates into ssq
}

// ssq is the plain sum of squares (rescaled == false) or sum (a 2^-e)^2, e = ilogb(maxabs).
__device__ __forceinline__ int shift_from_norm(double maxabs, double ssq, bool rescaled, double T) {
  if (row_nonfinite(maxabs, ssq)) return OZ2_SHIFT_NONFINITE;
  if (!(maxabs > 0.0)) return 0;  // zero row/column: all residues 0
  const double margin = 1.0 + 0x1p-30;
  double nm = sqrt(ssq * margin);
  int add = rescaled ? ilogb(maxabs) : 0;
  return choose_shift(nm, add, T);
}

// Exponent of the abs plane: maxabs * 2^sig in [64, 128).
__device__ __forceinline__ int abs_shift(double maxabs, int sh) {
  if (sh == OZ2_SHIFT_NONFINITE) return 0;
  if (!(maxabs > 0.0)) return OZ2_SHIFT_ZERO;
  return 6 - ilogb(maxabs);
}

__device__ __forceinline__ bool needs_rescale(double maxabs) {
  return maxabs > 0.0 && (maxabs > 0x1p400 || maxabs < 0x1p-400);
}

struct Dec {
  uint32_t c0, c1, c2;
  int E;
  bool neg;
};

__device__ __forceinline__ Dec decompose(double x) {  // x integer valued, |x| < 2^81
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
    E = min(E, OZ2_POW2_TABLE - 37);  // E <= 28 guaranteed; clamp for safety
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
  if (ri > (int)((p - 1) >> 1)) ri -= (int)p;  // p = 256: 128 -> -128
  return (int8_t)ri;
}

// Per-element prep work: scaled value, its truncation, the abs-plane value, and the
// contributions to ||dA||^2 and ||A'||^2.
struct Elem {
  Dec d;
  int8_t absv;
};
__device__ __forceinline__ Elem prep_elem(double a, int sh, int sig, double& da2, double& na2) {
  Elem e;
  if (sh == OZ2_SHIFT_NONFINITE) {
    e.d = decompose(0.0);
    e.absv = 0;
    return e;
  }
  const double xs = mul_pow2(a, sh);
  const double xt = trunc(xs);  // round toward zero (never rint)
  const double fr = xs - xt;    // exact
  da2 = fma(fr, fr, da2);
  na2 = fma(xt, xt, na2);
  e.d = decompose(xt);
  e.absv = (sig == OZ2_SHIFT_ZERO) ? (int8_t)0 : (int8_t)(int)fmin(127.0, trunc(mul_pow2(fabs(a), sig)));
  return e;
}

// float upper bound of sqrt(x) (x accumulated with relative error << 2^-30).
__device__ __forceinline__ int32_t norm_ub_bits(double x) {
  return __float_as_int(__double2float_ru(sqrt(x * (1.0 + 0x1p-30))));
}

// ------------------------------------------------------------------------------- prep kernel
// Workspace per problem: row_shift[4m] = {s_i | sig_i | da_i (float bits) | na_i (float bits)},
// col_shift[4n] likewise.  Planes 0..N-1: residues, plane N: abs plane.
__global__ void __launch_bounds__(THREADS) oz2_prep_kernel(const oz2_problem* __restrict__ probs, int a_blocks) {
  __shared__ uint8_t s_pow[GGEMM_MAX_MODULI][OZ2_POW2_TABLE];
  __shared__ __align__(16) int32_t s_tile[NPLANES][A_ROWS * A_KT / 4];
  __shared__ double s_max[THREADS / 32][32];
  __shared__ double s_ssq[THREADS / 32][32];
  __shared__ int s_shift[A_ROWS], s_sig[A_ROWS];

  const int tid = threadIdx.x;
  for (int i = tid; i < GGEMM_MAX_MODULI * OZ2_POW2_TABLE; i += THREADS)
    (&s_pow[0][0])[i] = c_tab.pow2[i / OZ2_POW2_TABLE][i % OZ2_POW2_TABLE];
  __syncthreads();

  const oz2_problem& pr = probs[blockIdx.y];
  const int N = pr.num_moduli;
  const int K = pr.k, Kp = pr.kp;
  if (N < 1 || N > GGEMM_MAX_MODULI) return;
  if (pr.alpha == 0.0) return;  // BLAS: alpha == 0 never references A, B (reconstruct outputs beta*C)
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
      const int sh = valid ? shift_from_norm(rowmax, S, resc, T) : 0;
      const int sg = valid ? abs_shift(rowmax, sh) : OZ2_SHIFT_ZERO;
      s_shift[tx] = sh;
      s_sig[tx] = sg;
      if (valid) {
        pr.row_shift[i] = sh;
        pr.row_shift[m + i] = sg;
      }
      if (sh == OZ2_SHIFT_NONFINITE) atomicAdd(&g_cnt[0], 1ull);
    }
    __syncthreads();
    const int sh = s_shift[tx], sg = s_sig[tx];

    double da2 = 0.0, na2 = 0.0;
    const int orow = tid >> 3, ow = tid & 7;  // write-out mapping: 32 rows x 8 words
    const bool ovalid = row0 + orow < m;
    for (int k0 = 0; k0 < Kp; k0 += A_KT) {
#pragma unroll
      for (int r = 0; r < A_KT / (THREADS / 32); ++r) {
        const int kk = ty + r * (THREADS / 32);
        const int k = k0 + kk;
        const double a = (valid && k < K) ? A[i + k * lda] : 0.0;
        const Elem el = prep_elem(a, sh, sg, da2, na2);
        const int boff = 4 * tile_word(tx, kk >> 2) + (kk & 3);
        for (int l = 0; l < N; ++l) reinterpret_cast<int8_t*>(s_tile[l])[boff] = residue_sym(el.d, l, s_pow);
        reinterpret_cast<int8_t*>(s_tile[N])[boff] = el.absv;
      }
      __syncthreads();
      if (ovalid) {
        int8_t* dst = pr.A8 + (size_t)(row0 + orow) * Kp + k0 + ow * 4;
        for (int l = 0; l <= N; ++l)
          *reinterpret_cast<int32_t*>(dst + l * pr.a8_mod_stride) =
              s_tile[l][tile_word(orow, ow)];
      }
      __syncthreads();
    }
    // ||dA(i,:)||, ||A'(i,:)|| upper bounds
    s_max[ty][tx] = da2;
    s_ssq[ty][tx] = na2;
    __syncthreads();
    if (ty == 0 && valid) {
      double D = 0.0, Nn = 0.0;
      for (int r = 0; r < THREADS / 32; ++r) { D += s_max[r][tx]; Nn += s_ssq[r][tx]; }
      pr.row_shift[2 * m + i] = norm_ub_bits(D);
      pr.row_shift[3 * m + i] = norm_ub_bits(Nn);
    }
  } else {
    // ------------------------------------------------ B: one warp per column (already K-contiguous)
    const int lane = tid & 31, w = tid >> 5;
    const int j = (blockIdx.x - a_blocks) * B_COLS + w;
    const int n = pr.n;
    if (j >= n) return;  // no __syncthreads below
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
    const int sg = abs_shift(mx, sh);
    if (lane == 0) {
      pr.col_shift[j] = sh;
      pr.col_shift[n + j] = sg;
      if (sh == OZ2_SHIFT_NONFINITE) atomicAdd(&g_cnt[0], 1ull);
    }

    double da2 = 0.0, na2 = 0.0;
    int8_t* base = pr.B8 + (size_t)j * Kp;
    for (int k0 = 4 * lane; k0 < Kp; k0 += 128) {
      Elem el[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int k = k0 + e;
        el[e] = prep_elem(k < K ? b[k] : 0.0, sh, sg, da2, na2);
      }
      for (int l = 0; l < N; ++l) {
        uint32_t packed = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) packed |= (uint32_t)(uint8_t)residue_sym(el[e].d, l, s_pow) << (8 * e);
        *reinterpret_cast<uint32_t*>(base + l * pr.b8_mod_stride + k0) = packed;
      }
      uint32_t packed = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) packed |= (uint32_t)(uint8_t)el[e].absv << (8 * e);
      *reinterpret_cast<uint32_t*>(base + N * pr.b8_mod_stride + k0) = packed;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      da2 += __shfl_xor_sync(0xffffffffu, da2, o);
      na2 += __shfl_xor_sync(0xffffffffu, na2, o);
    }
    if (lane == 0) {
      pr.col_shift[2 * n + j] = norm_ub_bits(da2);
      pr.col_shift[3 * n + j] = norm_ub_bits(na2);
    }
  }
}

// ------------------------------------------------------------------------ reconstruct kernel
__device__ __forceinline__ double fp64_dot(const oz2_problem& pr, int i, int j) {
  const double* a = pr.A + i;
  const double* b = pr.B + (size_t)j * pr.ldb;
  const size_t lda = (size_t)pr.lda;
  double sacc = 0.0;
  for (int k = 0; k < pr.k; ++k) sacc = fma(a[k * lda], b[k], sacc);
  return sacc;
}

__global__ void __launch_bounds__(THREADS) oz2_reconstruct_kernel(const oz2_problem* __restrict__ probs, int mode) {
  const oz2_problem& pr = probs[blockIdx.y];
  const int N = pr.num_moduli;
  if (N < 1 || N > GGEMM_MAX_MODULI) return;
  const int m = pr.m, n = pr.n, K = pr.k;
  const size_t total = (size_t)m * (size_t)n;
  const int32_t* __restrict__ C32 = pr.C32;
  const size_t cst = pr.c32_mod_stride;
  const double alpha = pr.alpha, beta = pr.beta;
  const double aalpha = fabs(alpha);
  const int64_t P1 = c_tab.P1[N];
  const double P2 = c_tab.P2[N], invPu = c_tab.inv_Pu[N];
  const int lu1 = c_tab.log2_u1[N];
  const double inv_u1 = ldexp(1.0, -lu1);
  const double crt_u = N * 255.0 * 0x1p-19 + 0x1p-10;  // CRT abs error in u1 units
  const double u = 0x1p-53;
  // gate: gamma_n, n = k (+1 alpha != 1) (+1 beta != 0); budget for the emulated value
  const double nt = (double)K + (alpha != 1.0 ? 1.0 : 0.0) + (beta != 0.0 ? 1.0 : 0.0);
  const double gam = (nt * u / (1.0 - nt * u) - 2.0 * u) * (1.0 - 0x1p-40);
  const int32_t* rs = pr.row_shift;
  const int32_t* cs = pr.col_shift;
  unsigned long long n_uncert = 0, n_repair = 0, n_safety = 0;

  for (size_t idx = (size_t)blockIdx.x * THREADS + threadIdx.x; idx < total; idx += (size_t)gridDim.x * THREADS) {
    const int i = (int)(idx % (size_t)m);
    const int j = (int)(idx / (size_t)m);
    double* cp = pr.C + i + (size_t)j * pr.ldc;
    const double c0 = (beta == 0.0) ? 0.0 : *cp;
    double out;
    if (alpha == 0.0) {  // BLAS semantics: A, B (and workspace) not referenced
      *cp = (beta == 0.0) ? 0.0 : beta * c0;
      continue;
    }
    const int rsh = rs[i], csh = cs[j];
    if (rsh == OZ2_SHIFT_NONFINITE || csh == OZ2_SHIFT_NONFINITE) {  // safety fallback, all modes
      const double d = fp64_dot(pr, i, j);
      out = (beta == 0.0) ? alpha * d : fma(alpha, d, beta * c0);
      ++n_safety;
      *cp = out;
      continue;
    }
    const int sig = rs[m + i], tau = cs[n + j];
    if (sig == OZ2_SHIFT_ZERO || tau == OZ2_SHIFT_ZERO) {  // A(i,:) or B(:,j) all zero: AB_ij = 0 exactly
      *cp = (beta == 0.0) ? alpha * 0.0 : fma(alpha, 0.0, beta * c0);  // NaN/Inf alpha propagates
      continue;
    }
    uint64_t S1 = 0;
    uint32_t S2 = 0;
    for (int l = 0; l < N; ++l) {
      const uint32_t p = (uint32_t)c_tab.p[l], mg = c_tab.magic[l];
      const int32_t c = C32[l * cst + idx];
      const uint32_t uc = c < 0 ? 0u - (uint32_t)c : (uint32_t)c;
      uint32_t r = mod_barrett(uc, p, mg);
      if (c < 0 && r) r = p - r;
      const uint32_t y = mod_barrett(r * (uint32_t)c_tab.qinv[N][l], p, mg);
      S1 += c_tab.s1[N][l] * (uint64_t)y;
      S2 += c_tab.s2[N][l] * y;
    }
    const double s2d = (double)S2 * (1.0 / (double)(1u << OZ2_S2_BITS));
    const double Q = rint(((double)S1 + s2d) * invPu);
    const int64_t X1 = (int64_t)S1 - (int64_t)Q * P1;
    const double xr = (double)X1 + (s2d - Q * P2);  // X / u1
    const int esc = lu1 - rsh - csh;
    const double v = mul_pow2(xr, esc);
    const double emu = (beta == 0.0) ? alpha * v : fma(alpha, v, beta * c0);

    if (mode == OZ2_REPAIR_OFF || (beta != 0.0 && !isfinite(c0))) {
      // OFF: plain emulation.  Non-finite C0 with beta != 0: result is non-finite whatever AB is;
      // propagate it exactly like BLAS instead of repairing.
      *cp = emu;
      continue;
    }
    // (i) rigorous error bound, u1 units, then output units
    const double da = __int_as_float(rs[2 * m + i]), na = __int_as_float(rs[3 * m + i]);
    const double db = __int_as_float(cs[2 * n + j]), nb = __int_as_float(cs[3 * n + j]);
    const double et = fma(da, nb, fma(na, db, da * db)) * (1.0 + 0x1p-40);
    const double Eu = (et * inv_u1 + crt_u + 4.0 * u * fabs(xr)) * (1.0 + 0x1p-40);
    const double E = mul_pow2(Eu, esc);
    // (ii) lower bound of (|A||B|)_ij
    const double L = (double)C32[N * cst + idx];
    double lab = mul_pow2(L, -(sig + tau));
    lab = fmax(lab, fabs(v) - E) * (1.0 - 0x1p-40);
    if (lab < 0x1p-1000) lab = 0.0;  // avoid relying on subnormal-rounded values
    // (iii) certificate (NaN/Inf anywhere -> not certified)
    const double lhs = aalpha * E * (1.0 + 0x1p-40);
    const double rhs = gam * (aalpha * lab + fabs(beta * c0)) * (1.0 - 0x1p-40);
    const bool cert = gam > 0.0 && E >= 0x1p-1000 && isfinite(lhs) && isfinite(v) && lhs <= rhs;
    if (cert) {
      out = emu;
    } else if (mode == OZ2_REPAIR_COUNT) {
      out = emu;
      ++n_uncert;
    } else {  // OZ2_REPAIR_ON: FP64 dot product, same gamma_n bound as native DGEMM
      const double d = fp64_dot(pr, i, j);
      out = (beta == 0.0) ? alpha * d : fma(alpha, d, beta * c0);
      ++n_repair;
    }
    *cp = out;
  }
  if (n_uncert) atomicAdd(&g_cnt[1], n_uncert);
  if (n_repair) atomicAdd(&g_cnt[2], n_repair);
  if (n_safety) atomicAdd(&g_cnt[3], n_safety);
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

int oz2_reconstruct_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s,
                            int repair_mode) {
  if (num <= 0) return 0;
  if (repair_mode != OZ2_REPAIR_OFF && repair_mode != OZ2_REPAIR_COUNT && repair_mode != OZ2_REPAIR_ON) return 1;
  if (num > 65535 || max_m < 0 || max_n < 0) return 1;
  if (int e = oz2_init_constants()) return e;
  const size_t mn = (size_t)max_m * (size_t)max_n;
  if (mn == 0) return 0;
  size_t gx = (mn + THREADS * 4 - 1) / (THREADS * 4);  // ~4 elements per thread
  if (gx > 1024) gx = 1024;
  oz2_reconstruct_kernel<<<dim3((unsigned)gx, num), THREADS, 0, s>>>(d, repair_mode);
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

// Componentwise planning: an entry is certified (no repair) roughly when
//   eps_N(k) * rho_ij <= gamma_k ~ k u,   rho_ij = ||a_i|| ||b_j|| / (|A||B|)_ij
// (the certificate uses the a-posteriori da/na norms, which are <= the a-priori ones, and a
// lower bound of (|A||B|)_ij, so this is a planning estimate, not a guarantee -- correctness
// never depends on it).  Choose the smallest N such that entries with rho_ij <= rho_budget are
// certified with a factor-2 margin.
int oz2_choose_num_moduli_cw(int k, double rho_budget) {
  if (k < 0 || k > OZ2_MAX_K) return -1;
  if (rho_budget < 1.0) rho_budget = 1.0;
  const double u = std::ldexp(1.0, -53);
  const double budget = 0.5 * (k > 2 ? (double)(k - 2) : 1.0) * u;
  for (int N = 2; N <= GGEMM_MAX_MODULI; ++N)
    if ((oz2_error_bound(k, N) - u) * rho_budget <= budget) return N;
  return GGEMM_MAX_MODULI;  // repairs keep the contract; N_max minimizes them
}

int oz2_read_counters(oz2_counters* out, int reset, cudaStream_t s) {
  unsigned long long h[4] = {0, 0, 0, 0};
  if (out) {
    if (cudaMemcpyFromSymbolAsync(h, g_cnt, sizeof(h), 0, cudaMemcpyDeviceToHost, s) != cudaSuccess) return 2;
  }
  if (reset) {
    void* p = nullptr;
    if (cudaGetSymbolAddress(&p, g_cnt) != cudaSuccess) return 4;
    if (cudaMemsetAsync(p, 0, sizeof(h), s) != cudaSuccess) return 4;  // ordered on s
  }
  if (out) {
    if (cudaStreamSynchronize(s) != cudaSuccess) return 3;
    out->nonfinite_vectors = h[0];
    out->uncertified = h[1];
    out->repaired = h[2];
    out->safety_fallback = h[3];
  }
  return 0;
}

int oz2_nonfinite_count(int* count, int reset, cudaStream_t s) {
  if (!count) return 1;
  oz2_counters c;
  int r = oz2_read_counters(&c, reset, s);
  *count = (int)c.nonfinite_vectors;
  return r;
}
