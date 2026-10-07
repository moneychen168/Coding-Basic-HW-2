// operand_reuse.h -- host pass that detects shared A / B operands across a group so the
// Ozaki-II prep (scale + INT8 residues) runs once per unique operand.  Owner: agent-O.
//
// Definitions
//   G            number of problems in the call.
//   N_g          moduli count the planner chose for problem g (num_moduli_per_problem[g]);
//                N_g <= 0 means "not emulated" (native / empty), such problems are ignored.
//   A-key(g)     (A pointer, lda, m, k) of problem g.  Two problems share A iff A-keys are equal.
//   B-key(g)     (B pointer, ldb, n, k) of problem g.
//   unique A u   one equivalence class of A-keys; rep = first problem index carrying it.
//
// IMPORTANT numerics constraint (checked against src/moduli.h + ozaki2_numerics.cu):
//   The moduli set IS a shared prefix (problem with N uses OZ2_MODULI[0..N)), BUT the per-row
//   scale shift is chosen from the 2-norm budget T_N (c_tab.T[N]).  So row_shift / col_shift and
//   the truncated integers A', B' depend on N.  Residues prepped with N_max are NOT valid for a
//   problem that runs with N < N_max (|A'B'| < P_N/2 would be violated).  Two correct policies:
//     REUSE_EXACT_N : N is part of the key; operands are shared only between problems with equal N.
//     REUSE_PROMOTE : operands are shared regardless of N; every problem is promoted to the max N
//                     of its connected component (problems linked through a shared A or B).
//                     The caller MUST run problem g with eff_nmod[g] moduli (prep, GEMM, recon).
//   If agent-N ever makes the shift N-independent (e.g. always use T_{GGEMM_MAX_MODULI} for the
//   scale), PROMOTE's ua_nmod/ub_nmod become "prep this many residues" and eff_nmod can stay N_g.
//
// Complexity O(G) expected (open-addressing hash with generation stamps + union-find).
// No allocation inside build_operand_reuse: all storage lives in reuse_arena, sized by reserve().
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggemm.h"

enum reuse_policy { REUSE_EXACT_N = 0, REUSE_PROMOTE = 1 };

enum {
  REUSE_OK = 0,
  REUSE_ERR_CAPACITY = 1,   // num > arena capacity (call reserve() at create time)
  REUSE_ERR_INVALID = 2     // null args / num < 0
};

// Non-owning view into a reuse_arena; valid until the next build on the same arena.
struct reuse_plan {
  int status;
  int num_problems;        // G
  int num_ua, num_ub;      // unique A / unique B counts
  int num_emu;             // problems with N_g > 0
  int num_promoted;        // problems with eff_nmod > N_g (PROMOTE only)
  const int* ua_rep;       // [num_ua] representative problem index (prep reads A from probs[rep])
  const int* ua_nmod;      // [num_ua] residues to prep for this A (max N over its users)
  const int* ub_rep;       // [num_ub]
  const int* ub_nmod;      // [num_ub]
  const int* prob_a;       // [G] unique-A id of problem g, -1 if N_g <= 0
  const int* prob_b;       // [G] unique-B id of problem g, -1 if N_g <= 0
  const int* eff_nmod;     // [G] moduli problem g must run with (== N_g under EXACT_N), 0 if ignored
};

struct reuse_arena {
  int cap = 0;                       // max problems
  uint32_t mask = 0;                 // hash table size - 1 (power of two >= 2*cap)
  uint32_t gen = 0;                  // generation stamp: slot live iff stamp == gen
  std::vector<uint32_t> ha_stamp, hb_stamp;
  std::vector<int> ha_id, hb_id;
  std::vector<int> ua_rep, ua_nmod, ub_rep, ub_nmod;
  std::vector<int> prob_a, prob_b, eff_nmod;
  std::vector<int> uf_parent, uf_max;   // nodes: [0,num_ua) = A, [cap, cap+num_ub) = B
  // Allocates everything. Call once (e.g. in ggemm_create) with max_problems.
  void reserve(int max_problems);
};

// The pass.  num_moduli_per_problem[g] <= 0 marks problem g as not emulated.
reuse_plan build_operand_reuse(reuse_arena& ws, const ggemm_problem_t* probs, int num,
                               const int* num_moduli_per_problem,
                               reuse_policy policy = REUSE_PROMOTE);

// Modelled prep traffic / workspace, from the formulas in the integration note (bytes).
//   prep_read   : 2 passes over FP64 A and B (norm pass + residue pass) = 16*(m*k + k*n)
//   prep_write  : residues N*(m*Kp + n*Kp) + shifts 4*(m + n)
//   ws_A8/ws_B8 : DESIGN [mod][prob] slab = nmax * sum(align256(rows*Kp))
// "before" = per-problem prep with N_g; "after" = per-unique prep with ua_nmod / ub_nmod.
struct reuse_bytes { double prep_read, prep_write, ws_A8, ws_B8; };
void operand_reuse_bytes(const reuse_plan& plan, const ggemm_problem_t* probs,
                         const int* num_moduli_per_problem, reuse_bytes* before, reuse_bytes* after);
