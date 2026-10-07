// moduli.h -- moduli set and CRT constants for Ozaki scheme II (INT8 residues).
// Owner: agent-N.
//
// All constants are computed on the host with a tiny exact big-integer helper
// (oz2_build_crt_tables) and then copied into __constant__ memory by
// ozaki2_numerics.cu.  Nothing here is hand-typed except the moduli themselves.
//
// Notation (used throughout ozaki2_numerics.cu):
//   N        number of moduli used by a problem (1..GGEMM_MAX_MODULI); always the
//            first N entries of OZ2_MODULI.
//   p_l      l-th modulus (<= 256, pairwise coprime).
//   P_N      = prod_{l<N} p_l.
//   M_l      = P_N / p_l (exact integer).
//   q_l      = M_l^{-1} mod p_l  (CRT inverse).
//   u1       = 2^{sh_N}, a power of two chosen so that every M_l / u1 < 2^49.
//   s1_l     = floor(M_l / u1)                      (exact uint64, < 2^49)
//   s2_l     = round((M_l / u1 - s1_l) * 2^18)      (uint32, <= 2^18)
//            so that M_l ~= u1 * (s1_l + s2_l * 2^-18) with abs error <= u1 * 2^-19.
//   P1       = floor(P_N / u1) (int64, < 2^57),  P2 = P_N/u1 - P1 in [0,1) (double).
//   T_N      = sqrt(P_N/2 * (1 - 2^-20)) * (1 - 2^-40): the 2-norm budget for one
//            scaled row of A' (and one column of B'), see ozaki2_numerics.cu.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef GGEMM_MAX_MODULI
#define GGEMM_MAX_MODULI 20
#endif

#define OZ2_POW2_TABLE 128   // 2^e mod p_l tabulated for e in [0, 128)
#define OZ2_S2_BITS 18       // fractional bits of s2_l

// First 16: the INT8 moduli set of Ozaki/Uchino/Imamura (Ozaki scheme II, eq. 19).
// 255 = 3*5*17, 253 = 11*23, 247 = 13*19, 217 = 7*31, others prime.
// Extended with the primes 181, 179, 173, 167 (coprime to all of the above).
// Pairwise coprimality is re-checked at table build time.
static const int OZ2_MODULI[GGEMM_MAX_MODULI] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223,
    217, 211, 199, 197, 193, 191, 181, 179, 173, 167};

struct oz2_crt_tables {
  // per modulus (independent of N)
  int32_t p[GGEMM_MAX_MODULI];
  uint32_t magic[GGEMM_MAX_MODULI];     // floor(2^32 / p_l), Barrett reduction
  uint8_t pow2[GGEMM_MAX_MODULI][OZ2_POW2_TABLE];  // 2^e mod p_l
  // per N (index N = 1..GGEMM_MAX_MODULI; entry 0 unused)
  uint64_t s1[GGEMM_MAX_MODULI + 1][GGEMM_MAX_MODULI];
  uint32_t s2[GGEMM_MAX_MODULI + 1][GGEMM_MAX_MODULI];
  int32_t qinv[GGEMM_MAX_MODULI + 1][GGEMM_MAX_MODULI];
  int64_t P1[GGEMM_MAX_MODULI + 1];
  double P2[GGEMM_MAX_MODULI + 1];
  double inv_Pu[GGEMM_MAX_MODULI + 1];  // u1 / P_N
  int32_t log2_u1[GGEMM_MAX_MODULI + 1];
  double T[GGEMM_MAX_MODULI + 1];       // row/col 2-norm budget
  double log2P[GGEMM_MAX_MODULI + 1];   // informational
};

// ---------------------------------------------------------------- host bigint
namespace oz2_detail {
struct BigU {
  std::vector<uint32_t> w;  // little endian limbs
  void trim() { while (!w.empty() && w.back() == 0) w.pop_back(); }
};
inline BigU big_from(uint64_t v) {
  BigU r; r.w.push_back((uint32_t)v); r.w.push_back((uint32_t)(v >> 32)); r.trim(); return r;
}
inline void big_mul_small(BigU& a, uint32_t m) {
  uint64_t c = 0;
  for (auto& x : a.w) { uint64_t t = (uint64_t)x * m + c; x = (uint32_t)t; c = t >> 32; }
  if (c) a.w.push_back((uint32_t)c);
}
inline uint32_t big_divmod_small(const BigU& a, uint32_t d, BigU* q) {
  uint64_t r = 0;
  BigU out; out.w.assign(a.w.size(), 0);
  for (size_t i = a.w.size(); i-- > 0;) {
    uint64_t cur = (r << 32) | a.w[i];
    out.w[i] = (uint32_t)(cur / d); r = cur % d;
  }
  out.trim();
  if (q) *q = out;
  return (uint32_t)r;
}
inline int big_bitlen(const BigU& a) {
  if (a.w.empty()) return 0;
  uint32_t top = a.w.back(); int b = 0; while (top) { ++b; top >>= 1; }
  return (int)(a.w.size() - 1) * 32 + b;
}
inline int big_bit(const BigU& a, int i) {
  if (i < 0) return 0;
  size_t li = (size_t)i / 32; if (li >= a.w.size()) return 0;
  return (a.w[li] >> (i % 32)) & 1;
}
inline BigU big_shr(const BigU& a, int s) {  // floor(a / 2^s), s >= 0
  BigU r; int nb = big_bitlen(a);
  for (int i = s; i < nb; ++i)
    if (big_bit(a, i)) { size_t j = (size_t)(i - s); if (r.w.size() <= j / 32) r.w.resize(j / 32 + 1, 0); r.w[j / 32] |= 1u << (j % 32); }
  r.trim(); return r;
}
inline BigU big_shl(const BigU& a, int s) {  // a * 2^s, s >= 0
  BigU r; int nb = big_bitlen(a);
  for (int i = 0; i < nb; ++i)
    if (big_bit(a, i)) { size_t j = (size_t)(i + s); if (r.w.size() <= j / 32) r.w.resize(j / 32 + 1, 0); r.w[j / 32] |= 1u << (j % 32); }
  r.trim(); return r;
}
inline BigU big_low(const BigU& a, int s) {  // a mod 2^s
  BigU r; for (int i = 0; i < s && i < big_bitlen(a); ++i)
    if (big_bit(a, i)) { if (r.w.size() <= (size_t)i / 32) r.w.resize(i / 32 + 1, 0); r.w[i / 32] |= 1u << (i % 32); }
  r.trim(); return r;
}
inline uint64_t big_to_u64(const BigU& a) {  // caller guarantees < 2^64
  uint64_t v = 0; for (size_t i = a.w.size(); i-- > 0;) v = (v << 32) | a.w[i]; return v;
}
inline double big_to_double(const BigU& a) {  // relative error ~1e-16, enough for margins
  long double v = 0; for (size_t i = a.w.size(); i-- > 0;) v = v * 4294967296.0L + a.w[i];
  return (double)v;
}
inline int gcd_i(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }
}  // namespace oz2_detail

// Returns 0 on success, nonzero if the moduli are not pairwise coprime or a
// range assumption is violated.
inline int oz2_build_crt_tables(oz2_crt_tables* t) {
  using namespace oz2_detail;
  std::memset(t, 0, sizeof(*t));
  for (int a = 0; a < GGEMM_MAX_MODULI; ++a)
    for (int b = a + 1; b < GGEMM_MAX_MODULI; ++b)
      if (gcd_i(OZ2_MODULI[a], OZ2_MODULI[b]) != 1) return 1;
  for (int l = 0; l < GGEMM_MAX_MODULI; ++l) {
    int p = OZ2_MODULI[l];
    if (p < 2 || p > 256) return 2;
    t->p[l] = p;
    t->magic[l] = (uint32_t)((1ull << 32) / (uint64_t)p);
    uint32_t v = 1 % p;
    for (int e = 0; e < OZ2_POW2_TABLE; ++e) { t->pow2[l][e] = (uint8_t)v; v = (v * 2) % p; }
  }
  for (int N = 1; N <= GGEMM_MAX_MODULI; ++N) {
    BigU P = big_from(1);
    for (int l = 0; l < N; ++l) big_mul_small(P, (uint32_t)OZ2_MODULI[l]);
    BigU M[GGEMM_MAX_MODULI];
    int maxbits = 0;
    for (int l = 0; l < N; ++l) {
      uint32_t rem = big_divmod_small(P, (uint32_t)OZ2_MODULI[l], &M[l]);
      if (rem != 0) return 3;
      int p = OZ2_MODULI[l];
      int mr = (int)big_divmod_small(M[l], (uint32_t)p, nullptr);
      int inv = -1;
      for (int x = 0; x < p; ++x) if ((mr * x) % p == 1 % p) { inv = x; break; }
      if (inv < 0) return 4;
      t->qinv[N][l] = inv;
      int b = big_bitlen(M[l]); if (b > maxbits) maxbits = b;
    }
    const int sh = maxbits - 49;  // u1 = 2^sh  =>  M_l / u1 < 2^49
    t->log2_u1[N] = sh;
    for (int l = 0; l < N; ++l) {
      if (sh >= 0) {
        t->s1[N][l] = big_to_u64(big_shr(M[l], sh));
        BigU low = big_low(M[l], sh);
        uint64_t s2;
        if (sh >= OZ2_S2_BITS) s2 = big_to_u64(big_shr(low, sh - OZ2_S2_BITS)) + (uint64_t)big_bit(low, sh - OZ2_S2_BITS - 1);
        else s2 = big_to_u64(low) << (OZ2_S2_BITS - sh);
        if (s2 > (1u << OZ2_S2_BITS)) return 5;
        t->s2[N][l] = (uint32_t)s2;
      } else {
        t->s1[N][l] = big_to_u64(big_shl(M[l], -sh));
        t->s2[N][l] = 0;
      }
      if (t->s1[N][l] >= (1ull << 49)) return 6;
    }
    BigU P1 = sh >= 0 ? big_shr(P, sh) : big_shl(P, -sh);
    if (big_bitlen(P1) > 57) return 7;
    t->P1[N] = (int64_t)big_to_u64(P1);
    t->P2[N] = sh > 0 ? std::ldexp(big_to_double(big_low(P, sh)), -sh) : 0.0;
    t->inv_Pu[N] = 1.0 / ((double)t->P1[N] + t->P2[N]);
    double Pd = big_to_double(P);
    t->log2P[N] = std::log2(Pd);
    t->T[N] = std::sqrt(Pd * 0.5 * (1.0 - std::ldexp(1.0, -20))) * (1.0 - std::ldexp(1.0, -40));
  }
  return 0;
}
