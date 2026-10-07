// operand_reuse.cpp -- see operand_reuse.h.  Owner: agent-O.  Host only.
#include "operand_reuse.h"

#include <algorithm>

namespace {

inline uint64_t mix64(uint64_t x) {   // splitmix64 finalizer
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27; x *= 0x94d049bb133111ebull;
  x ^= x >> 31; return x;
}
inline uint64_t key_hash(const void* p, int ld, int rows, int k, int n_or_zero) {
  uint64_t h = mix64((uint64_t)(uintptr_t)p);
  h = mix64(h ^ ((uint64_t)(uint32_t)ld << 32 | (uint32_t)rows));
  h = mix64(h ^ ((uint64_t)(uint32_t)k << 32 | (uint32_t)n_or_zero));
  return h;
}

inline int uf_find(int* par, int x) {
  while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }   // path halving
  return x;
}

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
inline size_t kp_of(int k) { return align_up((size_t)k, 32); }

}  // namespace

void reuse_arena::reserve(int max_problems) {
  if (max_problems < 1) max_problems = 1;
  cap = max_problems;
  uint32_t sz = 1;
  while (sz < 2u * (uint32_t)max_problems) sz <<= 1;
  mask = sz - 1;
  gen = 0;
  ha_stamp.assign(sz, 0); hb_stamp.assign(sz, 0);
  ha_id.assign(sz, -1);   hb_id.assign(sz, -1);
  ua_rep.assign(cap, -1); ua_nmod.assign(cap, 0);
  ub_rep.assign(cap, -1); ub_nmod.assign(cap, 0);
  prob_a.assign(cap, -1); prob_b.assign(cap, -1); eff_nmod.assign(cap, 0);
  uf_parent.assign(2 * (size_t)cap, 0); uf_max.assign(2 * (size_t)cap, 0);
}

reuse_plan build_operand_reuse(reuse_arena& ws, const ggemm_problem_t* probs, int num,
                               const int* nmod, reuse_policy policy) {
  reuse_plan r{};
  r.num_problems = num;
  if (num < 0 || (num > 0 && (!probs || !nmod))) { r.status = REUSE_ERR_INVALID; return r; }
  if (num > ws.cap) { r.status = REUSE_ERR_CAPACITY; return r; }

  if (++ws.gen == 0) {   // stamp wrap (once per 2^32 calls): O(table) clear, still no allocation
    std::fill(ws.ha_stamp.begin(), ws.ha_stamp.end(), 0u);
    std::fill(ws.hb_stamp.begin(), ws.hb_stamp.end(), 0u);
    ws.gen = 1;
  }
  const bool exact = (policy == REUSE_EXACT_N);
  const uint32_t mask = ws.mask, gen = ws.gen;
  int ua = 0, ub = 0, ne = 0;

  for (int g = 0; g < num; ++g) {
    const int N = nmod[g];
    if (N <= 0) { ws.prob_a[g] = ws.prob_b[g] = -1; ws.eff_nmod[g] = 0; continue; }
    ++ne;
    const ggemm_problem_t& p = probs[g];
    const int nkey = exact ? N : 0;

    // ---- A: key (A, lda, m, k[, N])
    uint32_t s = (uint32_t)key_hash(p.A, p.lda, p.m, p.k, nkey) & mask;
    int id = -1;
    for (;; s = (s + 1) & mask) {
      if (ws.ha_stamp[s] != gen) {          // empty slot -> new unique A
        id = ua++;
        ws.ha_stamp[s] = gen; ws.ha_id[s] = id;
        ws.ua_rep[id] = g; ws.ua_nmod[id] = N;
        break;
      }
      const int c = ws.ha_id[s];
      const ggemm_problem_t& q = probs[ws.ua_rep[c]];
      if (q.A == p.A && q.lda == p.lda && q.m == p.m && q.k == p.k && (!exact || ws.ua_nmod[c] == N)) {
        id = c; ws.ua_nmod[c] = std::max(ws.ua_nmod[c], N);
        break;
      }
    }
    ws.prob_a[g] = id;

    // ---- B: key (B, ldb, n, k[, N])
    s = (uint32_t)key_hash(p.B, p.ldb, p.n, p.k, nkey ^ 0x5a5a5a5a) & mask;
    for (;; s = (s + 1) & mask) {
      if (ws.hb_stamp[s] != gen) {
        id = ub++;
        ws.hb_stamp[s] = gen; ws.hb_id[s] = id;
        ws.ub_rep[id] = g; ws.ub_nmod[id] = N;
        break;
      }
      const int c = ws.hb_id[s];
      const ggemm_problem_t& q = probs[ws.ub_rep[c]];
      if (q.B == p.B && q.ldb == p.ldb && q.n == p.n && q.k == p.k && (!exact || ws.ub_nmod[c] == N)) {
        id = c; ws.ub_nmod[c] = std::max(ws.ub_nmod[c], N);
        break;
      }
    }
    ws.prob_b[g] = id;
    ws.eff_nmod[g] = N;
  }

  int promoted = 0;
  if (!exact && ne > 0) {
    // Union-find over the bipartite (unique A) -- problem -- (unique B) graph; every problem of a
    // connected component runs with the component max N, so all shared preps use one T_N.
    int* par = ws.uf_parent.data();
    int* mx = ws.uf_max.data();
    const int OB = ws.cap;   // B nodes start here
    for (int u = 0; u < ua; ++u) { par[u] = u; mx[u] = ws.ua_nmod[u]; }
    for (int u = 0; u < ub; ++u) { par[OB + u] = OB + u; mx[OB + u] = ws.ub_nmod[u]; }
    for (int g = 0; g < num; ++g) {
      if (ws.prob_a[g] < 0) continue;
      int x = uf_find(par, ws.prob_a[g]), y = uf_find(par, OB + ws.prob_b[g]);
      if (x == y) continue;
      if (x > y) std::swap(x, y);   // deterministic root
      par[y] = x; mx[x] = std::max(mx[x], mx[y]);
    }
    for (int u = 0; u < ua; ++u) ws.ua_nmod[u] = mx[uf_find(par, u)];
    for (int u = 0; u < ub; ++u) ws.ub_nmod[u] = mx[uf_find(par, OB + u)];
    for (int g = 0; g < num; ++g) {
      if (ws.prob_a[g] < 0) continue;
      const int N = ws.ua_nmod[ws.prob_a[g]];   // == ub_nmod[prob_b[g]] (same component)
      promoted += (N > ws.eff_nmod[g]);
      ws.eff_nmod[g] = N;
    }
  }

  r.status = REUSE_OK;
  r.num_ua = ua; r.num_ub = ub; r.num_emu = ne; r.num_promoted = promoted;
  r.ua_rep = ws.ua_rep.data(); r.ua_nmod = ws.ua_nmod.data();
  r.ub_rep = ws.ub_rep.data(); r.ub_nmod = ws.ub_nmod.data();
  r.prob_a = ws.prob_a.data(); r.prob_b = ws.prob_b.data(); r.eff_nmod = ws.eff_nmod.data();
  return r;
}

void operand_reuse_bytes(const reuse_plan& plan, const ggemm_problem_t* probs, const int* nmod,
                         reuse_bytes* before, reuse_bytes* after) {
  reuse_bytes b{}, a{};
  int nmax_b = 0;
  double SA = 0, SB = 0;
  for (int g = 0; g < plan.num_problems; ++g) {
    if (nmod[g] <= 0) continue;
    const ggemm_problem_t& p = probs[g];
    const double kp = (double)kp_of(p.k);
    b.prep_read += 16.0 * ((double)p.m * p.k + (double)p.k * p.n);
    b.prep_write += nmod[g] * kp * ((double)p.m + p.n) + 4.0 * ((double)p.m + p.n);
    SA += (double)align_up((size_t)p.m * kp_of(p.k), 256);
    SB += (double)align_up((size_t)p.n * kp_of(p.k), 256);
    nmax_b = std::max(nmax_b, nmod[g]);
  }
  b.ws_A8 = nmax_b * SA; b.ws_B8 = nmax_b * SB;

  int nmax_a = 0;
  double SA2 = 0, SB2 = 0;
  for (int u = 0; u < plan.num_ua; ++u) {
    const ggemm_problem_t& p = probs[plan.ua_rep[u]];
    const double kp = (double)kp_of(p.k);
    a.prep_read += 16.0 * (double)p.m * p.k;
    a.prep_write += plan.ua_nmod[u] * kp * p.m + 4.0 * p.m;
    SA2 += (double)align_up((size_t)p.m * kp_of(p.k), 256);
    nmax_a = std::max(nmax_a, plan.ua_nmod[u]);
  }
  for (int u = 0; u < plan.num_ub; ++u) {
    const ggemm_problem_t& p = probs[plan.ub_rep[u]];
    const double kp = (double)kp_of(p.k);
    a.prep_read += 16.0 * (double)p.k * p.n;
    a.prep_write += plan.ub_nmod[u] * kp * p.n + 4.0 * p.n;
    SB2 += (double)align_up((size_t)p.n * kp_of(p.k), 256);
    nmax_a = std::max(nmax_a, plan.ub_nmod[u]);
  }
  a.ws_A8 = nmax_a * SA2; a.ws_B8 = nmax_a * SB2;
  if (before) *before = b;
  if (after) *after = a;
}
