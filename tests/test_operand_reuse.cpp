// Host-only unit tests for src/operand_reuse.{h,cpp} (agent-O).
// Pointers are fake host addresses (never dereferenced).  Build:
//   g++ -std=c++17 -O2 -I include -I src -I /usr/local/cuda-13.0/include \
//       tests/test_operand_reuse.cpp src/operand_reuse.cpp -o build/test_operand_reuse
#include <cstdio>
#include <vector>

#include "operand_reuse.h"

static int g_fail = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
  } while (0)

static const double* P(uintptr_t x) { return reinterpret_cast<const double*>(x * 4096); }

static ggemm_problem_t mk(const double* A, int lda, const double* B, int ldb, int m, int n, int k) {
  ggemm_problem_t p{};
  p.m = m; p.n = n; p.k = k; p.A = A; p.lda = lda; p.B = B; p.ldb = ldb;
  p.C = nullptr; p.ldc = m; p.alpha = 1.0; p.beta = 0.0;
  return p;
}

static void test_no_sharing(reuse_arena& ws) {
  std::vector<ggemm_problem_t> pr;
  std::vector<int> nm;
  for (int g = 0; g < 50; ++g) { pr.push_back(mk(P(1000 + g), 64, P(2000 + g), 64, 64, 32, 64)); nm.push_back(6); }
  reuse_plan r = build_operand_reuse(ws, pr.data(), 50, nm.data());
  CHECK(r.status == REUSE_OK);
  CHECK(r.num_ua == 50 && r.num_ub == 50 && r.num_promoted == 0);
  for (int g = 0; g < 50; ++g) CHECK(r.prob_a[g] == g && r.prob_b[g] == g && r.eff_nmod[g] == 6);
  reuse_bytes b, a;
  operand_reuse_bytes(r, pr.data(), nm.data(), &b, &a);
  CHECK(b.prep_read == a.prep_read && b.prep_write == a.prep_write && b.ws_A8 == a.ws_A8);
}

static void test_all_share_A(reuse_arena& ws) {   // tensor-parallel: one A, B split in column blocks
  std::vector<ggemm_problem_t> pr;
  std::vector<int> nm;
  for (int g = 0; g < 8; ++g) { pr.push_back(mk(P(1), 512, P(100 + g), 256, 512, 128 + g, 256)); nm.push_back(7); }
  reuse_plan r = build_operand_reuse(ws, pr.data(), 8, nm.data());
  CHECK(r.num_ua == 1 && r.num_ub == 8);
  CHECK(r.ua_rep[0] == 0 && r.ua_nmod[0] == 7);
  for (int g = 0; g < 8; ++g) CHECK(r.prob_a[g] == 0 && r.prob_b[g] == g && r.eff_nmod[g] == 7);
  reuse_bytes b, a;
  operand_reuse_bytes(r, pr.data(), nm.data(), &b, &a);
  // A prepped once instead of 8 times: saves 7 * (16*m*k + N*m*Kp + 4*m)
  const double m = 512, k = 256, kp = 256;
  CHECK(b.prep_read - a.prep_read == 7 * 16 * m * k);
  CHECK(b.prep_write - a.prep_write == 7 * (7 * m * kp + 4 * m));
  CHECK(b.ws_A8 == 8 * a.ws_A8);
}

static void test_lora(reuse_arena& ws) {
  // multi-LoRA shrink: 2 token batches X0,X1, each times 3 adapters; adapter weights shared too.
  std::vector<ggemm_problem_t> pr;
  std::vector<int> nm;
  for (int x = 0; x < 2; ++x)
    for (int a = 0; a < 3; ++a) { pr.push_back(mk(P(10 + x), 128, P(50 + a), 4096, 128, 16, 4096)); nm.push_back(8); }
  reuse_plan r = build_operand_reuse(ws, pr.data(), 6, nm.data());
  CHECK(r.num_ua == 2 && r.num_ub == 3);
  for (int g = 0; g < 6; ++g) { CHECK(r.prob_a[g] == g / 3); CHECK(r.prob_b[g] == g % 3); }
  CHECK(r.ua_rep[1] == 3 && r.ub_rep[2] == 2);
}

static void test_same_ptr_diff_ld(reuse_arena& ws) {
  std::vector<ggemm_problem_t> pr = {
      mk(P(7), 256, P(8), 64, 128, 32, 64),
      mk(P(7), 512, P(8), 128, 128, 32, 64),   // same pointers, different lda / ldb
      mk(P(7), 256, P(8), 64, 64, 32, 64),     // same A ptr+lda, different m -> not shared
      mk(P(7), 256, P(8), 64, 128, 32, 32),    // different k -> neither shared
      mk(P(7), 256, P(8), 64, 128, 16, 64)};   // same A key as #0, different n -> A shared, B not
  std::vector<int> nm(pr.size(), 5);
  reuse_plan r = build_operand_reuse(ws, pr.data(), (int)pr.size(), nm.data());
  CHECK(r.num_ua == 4 && r.num_ub == 4);
  CHECK(r.prob_a[0] != r.prob_a[1] && r.prob_b[0] != r.prob_b[1]);
  CHECK(r.prob_a[0] != r.prob_a[2] && r.prob_b[0] == r.prob_b[2]);
  CHECK(r.prob_a[0] != r.prob_a[3] && r.prob_b[0] != r.prob_b[3]);
  CHECK(r.prob_a[0] == r.prob_a[4] && r.prob_b[0] != r.prob_b[4]);
}

static void test_diff_N(reuse_arena& ws) {
  // P0:(A0,B0,N=3)  P1:(A0,B1,N=5)  P2:(A1,B1,N=2)  P3:(A2,B2,N=4)  P4: native (N=0)
  std::vector<ggemm_problem_t> pr = {
      mk(P(1), 64, P(11), 64, 64, 64, 64), mk(P(1), 64, P(12), 64, 64, 64, 64),
      mk(P(2), 64, P(12), 64, 64, 64, 64), mk(P(3), 64, P(13), 64, 64, 64, 64),
      mk(P(1), 64, P(11), 64, 64, 64, 64)};
  std::vector<int> nm = {3, 5, 2, 4, 0};
  // PROMOTE: A0 needs max(3,5)=5; component {P0,P1,P2} all promoted to 5; P3 stays 4.
  reuse_plan r = build_operand_reuse(ws, pr.data(), 5, nm.data(), REUSE_PROMOTE);
  CHECK(r.num_ua == 3 && r.num_ub == 3 && r.num_emu == 4);
  CHECK(r.ua_nmod[r.prob_a[0]] == 5 && r.ua_nmod[r.prob_a[2]] == 5 && r.ua_nmod[r.prob_a[3]] == 4);
  CHECK(r.ub_nmod[r.prob_b[0]] == 5 && r.ub_nmod[r.prob_b[1]] == 5 && r.ub_nmod[r.prob_b[3]] == 4);
  CHECK(r.eff_nmod[0] == 5 && r.eff_nmod[1] == 5 && r.eff_nmod[2] == 5 && r.eff_nmod[3] == 4);
  CHECK(r.num_promoted == 2);
  CHECK(r.prob_a[4] == -1 && r.prob_b[4] == -1 && r.eff_nmod[4] == 0);
  // EXACT_N: different N never share; eff_nmod == N_g.
  reuse_plan e = build_operand_reuse(ws, pr.data(), 5, nm.data(), REUSE_EXACT_N);
  CHECK(e.num_ua == 4 && e.num_ub == 4 && e.num_promoted == 0);
  CHECK(e.prob_a[0] != e.prob_a[1]);
  for (int g = 0; g < 4; ++g) CHECK(e.eff_nmod[g] == nm[g]);
  // EXACT_N with equal N does share.
  std::vector<int> eq = {5, 5, 2, 4, 0};
  e = build_operand_reuse(ws, pr.data(), 5, eq.data(), REUSE_EXACT_N);
  CHECK(e.prob_a[0] == e.prob_a[1] && e.ua_nmod[e.prob_a[0]] == 5);
}

static void test_capacity_and_reuse_arena(reuse_arena& ws) {
  std::vector<ggemm_problem_t> pr(ws.cap + 1, mk(P(1), 8, P(2), 8, 8, 8, 8));
  std::vector<int> nm(pr.size(), 2);
  CHECK(build_operand_reuse(ws, pr.data(), (int)pr.size(), nm.data()).status == REUSE_ERR_CAPACITY);
  // repeated calls on the same arena must not see stale hash entries (generation stamps)
  for (int it = 0; it < 3; ++it) {
    reuse_plan r = build_operand_reuse(ws, pr.data(), ws.cap, nm.data());
    CHECK(r.num_ua == 1 && r.num_ub == 1);
  }
  const int* before = ws.ha_id.data();
  build_operand_reuse(ws, pr.data(), ws.cap, nm.data());
  CHECK(ws.ha_id.data() == before);   // no reallocation
}

int main() {
  reuse_arena ws;
  ws.reserve(64);
  test_no_sharing(ws);
  test_all_share_A(ws);
  test_lora(ws);
  test_same_ptr_diff_ld(ws);
  test_diff_N(ws);
  test_capacity_and_reuse_arena(ws);
  std::printf(g_fail ? "operand_reuse: %d FAILED\n" : "operand_reuse: all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
