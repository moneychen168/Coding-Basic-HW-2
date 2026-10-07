// tests/test_grouped.cu -- correctness tests for every ggemm path (owner: agent-B).
// Depends only on include/ggemm.h (+ bench/workloads, tests/reference).
//
// Gate (never relaxed): for every non-empty problem g,
//     max_ij |C_ij - Cref_ij| / den_ij  <=  gamma_{n_g},  gamma_n = n u/(1-n u), u = 2^-53,
//     n_g = k_g (+1 if alpha != 1) (+1 if beta != 0),  den = |alpha||A||B| + |beta||C0|,
// i.e. the classical componentwise error bound that native FP64 GEMM itself satisfies
// (Higham 2002, Sec. 3.5). An emulated "FP64-level" GEMM must meet the same contract.
// It is applied to native paths, to emulation with adaptive moduli (num_moduli=0,
// target 1e-15) and to AUTO. Fixed-N emulation sweeps are accuracy measurements, not
// contract claims, so they are reported by the bench, not gated here.
//
// Paths ("all"): native_loop, native_grouped, emu_loop, emu_grouped, auto, emu_grouped_reuse,
//   emu_loop_batched.
// Usage: test_grouped [--set test|quick] [--paths all|a,b] [--phis 0,0.5,1,2] [--csv out.csv]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "ggemm.h"
#include "../bench/workloads.h"
#include "reference.h"

static int g_pass = 0, g_fail = 0;
static FILE* g_csv = nullptr;

#define CHECK(cond, ...) do { if (cond) { ++g_pass; } else { ++g_fail; \
  printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)

static ggemm_config_t cfg_for(ggemm_path_t p) {
  ggemm_config_t c;
  c.num_moduli = 0;              // adaptive
  c.target_rel_err = 1e-15;      // FP64 level
  c.allow_fallback = (p == GGEMM_PATH_AUTO) ? 1 : 0;
  return c;
}
static bool is_emu(ggemm_path_t p) {
  return ggemm_path_is_emu(p);   // incl. emu_grouped_reuse, emu_loop_batched
}

// Gate per repair mode (see ggemm_expected_repair_mode):
//   strict (native, REPAIR_ON, or certificate count unavailable): every entry err_ij <= gamma_n den_ij
//   REPAIR_COUNT: #entries with err_ij > gamma_n den_ij (non-finite included) <= stats.num_uncertified,
//                 i.e. the certificate may over-report but must never miss a violating entry.
static bool gate_ok(ggemm_path_t path, const ErrStats& e, const ggemm_stats_t& st, const char** kind) {
  long long unc = stats_num_uncertified(st);
  if (ggemm_expected_repair_mode(path) == REPAIR_COUNT && unc >= 0) {
    *kind = "cert";
    return e.n_over_gamma <= (unsigned long long)unc;
  }
  *kind = "strict";
  return e.nonfinite == 0 && e.ratio_gamma <= 1.0;
}

// Runs one path on b (C reset to C0 first), compares to reference, applies the gate.
static void run_and_check(const char* tag, ggemm_path_t path, const ggemm_config_t& cfg,
                          WorkloadBuffers& b, RefBuffers& ref, double native_cw, bool gate) {
  ggemm_handle_t h = nullptr;
  int G = (int)b.probs.size();
  int rc = ggemm_create(&h, G > 0 ? G : 1, b.nA + 1, b.nB + 1, b.nC + 1, 0);
  CHECK(rc == 0 && h, "%s %s: ggemm_create rc=%d", tag, ggemm_path_name(path), rc);
  if (rc || !h) return;
  reset_C(&b);
  cudaDeviceSynchronize();
  ggemm_stats_t st; memset(&st, 0xff, sizeof st);
  rc = ggemm_grouped_dgemm(h, path, &cfg, b.probs.data(), G, &st);
  cudaError_t ce = cudaDeviceSynchronize();
  CHECK(rc == 0 && ce == cudaSuccess, "%s %s: rc=%d cuda=%s", tag, ggemm_path_name(path), rc,
        cudaGetErrorString(ce));
  if (rc == 0 && ce == cudaSuccess) {
    ErrStats e; ref_compare(b, &ref, &e);
    double vs_native = native_cw > 0 ? e.err_cw / native_cw : (e.err_cw == 0 ? 1.0 : INFINITY);
    const char* kind = "";
    bool ok = gate_ok(path, e, st, &kind);
    long long unc = stats_num_uncertified(st), rep = stats_num_repaired(st);
    printf("  %-5s %-17s %-14s err_cw=%.3e  err/gamma=%.3e  over_gamma=%llu uncert=%lld repaired=%lld "
           "gate=%s  vs_native=%8.3g  rel_max=%.3e  fallback=%d launches=%d moduli=[%d,%d]\n",
           ok ? "ok" : (gate ? "FAIL" : "info"), ggemm_path_name(path), tag, e.err_cw, e.ratio_gamma,
           e.n_over_gamma, unc, rep, kind, vs_native, e.rel_max, st.num_fallback, st.kernel_launches,
           st.moduli_min, st.moduli_max);
    if (gate) CHECK(ok, "%s %s [%s]: over_gamma=%llu uncertified=%lld; worst problem %d (m=%d n=%d k=%d), "
                    "nonfinite=%llu", tag, ggemm_path_name(path), kind, e.n_over_gamma, unc,
                    ggemm_path_name(path), e.worst_problem,
                    e.worst_problem >= 0 ? b.probs[e.worst_problem].m : -1,
                    e.worst_problem >= 0 ? b.probs[e.worst_problem].n : -1,
                    e.worst_problem >= 0 ? b.probs[e.worst_problem].k : -1, e.nonfinite);
    CHECK(st.num_problems == G, "%s %s: stats.num_problems=%d expected %d", tag, ggemm_path_name(path),
          st.num_problems, G);
    CHECK(st.num_fallback >= 0 && st.num_fallback <= G, "%s: num_fallback=%d out of range", tag, st.num_fallback);
    if (is_emu(path) && !cfg.allow_fallback)
      CHECK(st.num_fallback == 0, "%s %s: allow_fallback=0 but num_fallback=%d", tag, ggemm_path_name(path),
            st.num_fallback);
    int nonempty = 0;
    for (auto& q : b.probs) nonempty += (q.m > 0 && q.n > 0);
    // moduli range is only meaningful if >= 1 non-empty problem was actually emulated
    if (is_emu(path) && nonempty - st.num_fallback >= 1)
      CHECK(st.moduli_min >= 2 && st.moduli_max <= GGEMM_MAX_MODULI && st.moduli_min <= st.moduli_max,
            "%s %s: moduli range [%d,%d]", tag, ggemm_path_name(path), st.moduli_min, st.moduli_max);
    if (g_csv)
      fprintf(g_csv, "%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%llu,%d,%d,%d,%d,%s,%llu,%lld,%lld,%s\n", tag,
              ggemm_path_name(path), G, e.err_cw, e.ratio_gamma, vs_native, e.rel_max, e.nonfinite,
              st.num_fallback, st.kernel_launches, st.moduli_min, st.moduli_max, (!gate || ok) ? "ok" : "FAIL",
              e.n_over_gamma, unc, rep, kind);
  }
  ggemm_destroy(h);
}

// Full matrix: workload x phi x (alpha,beta) x path.
static void test_workloads(const std::string& set, const std::vector<ggemm_path_t>& paths,
                           const std::vector<double>& phis, cublasHandle_t cub) {
  struct AB { double a, b; };
  for (auto& w : make_workloads(set)) {
    int pad = (w.name.find("7x13x33") != std::string::npos) ? 3 : 0;   // also exercise lda > m
    for (double phi : phis) {
      std::vector<AB> abs = {{1.0, 0.0}};
      if (phi == 1.0) abs.push_back({1.5, -0.75});
      for (auto ab : abs) {
        WorkloadBuffers b;
        if (alloc_workload(w, phi, ab.a, ab.b, pad, &b)) { CHECK(false, "alloc %s", w.name.c_str()); continue; }
        RefBuffers ref; ref_alloc(&ref, b); ref_compute(b, &ref);
        char tag[160];
        snprintf(tag, sizeof tag, "%s/phi%.1f/a%.2g_b%.2g", w.name.c_str(), phi, ab.a, ab.b);
        printf("[%s] G=%zu max_k=%d flops=%.3e\n", tag, w.shapes.size(), w.max_k(), w.total_flops());
        // independent cuBLAS DGEMM baseline (same data) -> denominator of vs_native
        reset_C(&b);
        run_cublas_dgemm_loop(b, cub);
        ErrStats en; ref_compare(b, &ref, &en);
        printf("  %-5s %-15s %-14s err_cw=%.3e  err/gamma=%.3e\n", en.ratio_gamma <= 1 ? "ok" : "FAIL",
               "cublas_dgemm", "", en.err_cw, en.ratio_gamma);
        CHECK(en.ratio_gamma <= 1.0 && en.nonfinite == 0, "%s cublas reference sanity", tag);
        for (auto p : paths) run_and_check(tag, p, cfg_for(p), b, ref, en.err_cw, true);
        ref_free(&ref); free_workload(&b);
      }
    }
  }
}

static Workload custom(const char* name, std::vector<Shape> s) {
  Workload w; w.name = name; w.family = "edge"; w.seed = 12345 + strlen(name); w.shapes = s; return w;
}

static void test_edges(const std::vector<ggemm_path_t>& paths, cublasHandle_t cub) {
  printf("[edge cases]\n");
  // 1. num_problems = 0 (also with problems == nullptr)
  for (auto p : paths) {
    ggemm_handle_t h = nullptr;
    int rc = ggemm_create(&h, 4, 64, 64, 64, 0);
    CHECK(rc == 0, "create");
    if (rc) continue;
    ggemm_config_t c = cfg_for(p);
    ggemm_stats_t st; memset(&st, 0xff, sizeof st);
    rc = ggemm_grouped_dgemm(h, p, &c, nullptr, 0, &st);
    CHECK(rc == 0, "%s: num_problems=0 rc=%d", ggemm_path_name(p), rc);
    CHECK(rc != 0 || (st.num_problems == 0 && st.num_fallback == 0), "%s: num_problems=0 stats %d/%d",
          ggemm_path_name(p), st.num_problems, st.num_fallback);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "%s: num_problems=0 cuda error", ggemm_path_name(p));
    ggemm_destroy(h);
  }
  // 2..4: run through the standard gate
  struct Case { Workload w; double a, b; bool nan_c; };
  std::vector<Case> cases = {
      {custom("one_100x70x50", {{100, 70, 50}}), 1.0, 0.0, false},
      {custom("one_1x1x1", {{1, 1, 1}}), 1.0, 0.0, false},
      {custom("zero_dims_mix", {{0, 32, 32}, {32, 0, 32}, {32, 32, 0}, {48, 40, 36}, {0, 0, 0}}), 1.5, -0.5, false},
      {custom("all_empty", {{0, 16, 16}, {16, 0, 16}}), 1.0, 0.0, false},
      {custom("beta0_nanC", {{40, 24, 72}, {17, 33, 5}}), 1.0, 0.0, true},   // beta=0 must not read C
      {custom("k_not_mult32", {{33, 31, 1}, {65, 3, 31}, {5, 129, 97}}), -2.0, 1.0, false},
  };
  for (auto& cs : cases) {
    WorkloadBuffers b;
    if (alloc_workload(cs.w, 1.0, cs.a, cs.b, 0, &b)) { CHECK(false, "alloc %s", cs.w.name.c_str()); continue; }
    if (cs.nan_c) { fill_const_device(b.dC0, b.nC, NAN); cudaDeviceSynchronize(); }
    RefBuffers ref; ref_alloc(&ref, b); ref_compute(b, &ref);
    reset_C(&b); run_cublas_dgemm_loop(b, cub);
    ErrStats en; ref_compare(b, &ref, &en);
    std::string tag = "edge/" + cs.w.name;
    for (auto p : paths) run_and_check(tag.c_str(), p, cfg_for(p), b, ref, en.err_cw, true);
    ref_free(&ref); free_workload(&b);
  }
  // 5. fallback accounting: k > 131072 cannot be done by INT8/INT32 accumulation directly.
  {
    Workload w = custom("bigk_fallback", {{4, 4, 140000}, {64, 64, 64}, {8, 8, 8}});
    WorkloadBuffers b;
    if (!alloc_workload(w, 0.5, 1.0, 0.0, 0, &b)) {
      RefBuffers ref; ref_alloc(&ref, b); ref_compute(b, &ref);
      reset_C(&b); run_cublas_dgemm_loop(b, cub);
      ErrStats en; ref_compare(b, &ref, &en);
      // AUTO with fallback allowed: must succeed, be correct, and report >= 1 fallback.
      {
        ggemm_handle_t h; ggemm_create(&h, 3, b.nA + 1, b.nB + 1, b.nC + 1, 0);
        ggemm_config_t c = cfg_for(GGEMM_PATH_AUTO);
        ggemm_stats_t st; memset(&st, 0, sizeof st);
        reset_C(&b);
        int rc = ggemm_grouped_dgemm(h, GGEMM_PATH_AUTO, &c, b.probs.data(), 3, &st);
        cudaDeviceSynchronize();
        CHECK(rc == 0, "AUTO bigk rc=%d", rc);
        CHECK(st.num_fallback >= 1, "AUTO k=140000 must fall back, num_fallback=%d", st.num_fallback);
        ErrStats e; ref_compare(b, &ref, &e);
        const char* kind;
        CHECK(rc != 0 || gate_ok(GGEMM_PATH_AUTO, e, st, &kind), "AUTO bigk err/gamma=%.3e", e.ratio_gamma);
        printf("  bigk AUTO fallback=%d err/gamma=%.3e\n", st.num_fallback, e.ratio_gamma);
        ggemm_destroy(h);
      }
      // allow_fallback=0: either refuse (nonzero rc) or compute correctly with 0 fallbacks.
      for (auto p : {GGEMM_PATH_EMU_GROUPED, GGEMM_PATH_AUTO}) {
        ggemm_handle_t h; ggemm_create(&h, 3, b.nA + 1, b.nB + 1, b.nC + 1, 0);
        ggemm_config_t c = cfg_for(p); c.allow_fallback = 0;
        ggemm_stats_t st; memset(&st, 0, sizeof st);
        reset_C(&b);
        int rc = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), 3, &st);
        cudaDeviceSynchronize();
        if (rc == 0) {
          ErrStats e; ref_compare(b, &ref, &e);
          const char* kind;
          CHECK(st.num_fallback == 0 && gate_ok(p, e, st, &kind), "%s bigk no-fallback [%s]: fb=%d err/gamma=%.3e "
                "over_gamma=%llu uncert=%lld", ggemm_path_name(p), kind, st.num_fallback, e.ratio_gamma,
                e.n_over_gamma, stats_num_uncertified(st));
        } else {
          ++g_pass;
          printf("  bigk %s allow_fallback=0 -> refused (rc=%d), accepted\n", ggemm_path_name(p), rc);
        }
        ggemm_destroy(h);
      }
      ref_free(&ref); free_workload(&b);
    } else CHECK(false, "alloc bigk");
  }
  // 6. capacity / argument violations must return nonzero, never allocate or crash.
  {
    Workload w = custom("cap3", {{16, 16, 16}, {16, 16, 16}, {16, 16, 16}});
    WorkloadBuffers b; alloc_workload(w, 0.0, 1.0, 0.0, 0, &b);
    for (auto p : paths) {
      ggemm_handle_t h; ggemm_create(&h, 2, 4096, 4096, 4096, 0);
      ggemm_config_t c = cfg_for(p); ggemm_stats_t st;
      int rc = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), 3, &st);
      CHECK(rc != 0, "%s: 3 problems > max_problems=2 must fail", ggemm_path_name(p));
      ggemm_destroy(h);
      ggemm_create(&h, 8, 100, 100, 100, 0);   // 3*256 elems > 100
      rc = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), 3, &st);
      CHECK(rc != 0, "%s: total elems > max_total_*_elems must fail", ggemm_path_name(p));
      ggemm_destroy(h);
    }
    for (auto p : {GGEMM_PATH_EMU_GROUPED, GGEMM_PATH_AUTO}) {
      ggemm_handle_t h; ggemm_create(&h, 8, 4096, 4096, 4096, 0);
      ggemm_stats_t st;
      for (int bad : {1, GGEMM_MAX_MODULI + 1, -1}) {
        ggemm_config_t c = cfg_for(p); c.num_moduli = bad;
        int rc = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), 3, &st);
        CHECK(rc != 0, "%s: num_moduli=%d must be rejected", ggemm_path_name(p), bad);
      }
      ggemm_destroy(h);
    }
    cudaDeviceSynchronize();
    free_workload(&b);
  }
  // 7. O(1) launches for EMU_GROUPED: launch count must not grow with G (DESIGN: 3 kernels).
  {
    int launches[2] = {-1, -1}; int Gs[2] = {16, 1024};
    for (int t = 0; t < 2; ++t) {
      Workload w = custom("launch", std::vector<Shape>(Gs[t], Shape{32, 32, 32}));
      WorkloadBuffers b; alloc_workload(w, 0.5, 1.0, 0.0, 0, &b);
      ggemm_handle_t h; ggemm_create(&h, Gs[t], b.nA, b.nB, b.nC, 0);
      ggemm_config_t c = cfg_for(GGEMM_PATH_EMU_GROUPED); ggemm_stats_t st; memset(&st, 0, sizeof st);
      if (ggemm_grouped_dgemm(h, GGEMM_PATH_EMU_GROUPED, &c, b.probs.data(), Gs[t], &st) == 0)
        launches[t] = st.kernel_launches;
      cudaDeviceSynchronize(); ggemm_destroy(h); free_workload(&b);
    }
    printf("  emu_grouped launches: G=16 -> %d, G=1024 -> %d\n", launches[0], launches[1]);
    CHECK(launches[0] > 0 && launches[0] == launches[1], "EMU_GROUPED launches depend on G (%d vs %d)",
          launches[0], launches[1]);
  }
  // 8. bitwise reproducibility: two calls on identical input give identical bytes.
  {
    auto tw = make_workloads("test");
    Workload w = tw.size() > 5 ? tw[5] : tw[0];   // MoE prefill (skewed, unequal m_e)
    WorkloadBuffers b; alloc_workload(w, 1.0, 1.0, 0.0, 0, &b);
    for (auto p : paths) {
      ggemm_handle_t h; ggemm_create(&h, (int)b.probs.size(), b.nA, b.nB, b.nC, 0);
      ggemm_config_t c = cfg_for(p); ggemm_stats_t st;
      std::vector<double> r1(b.nC), r2(b.nC);
      int rc1 = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), (int)b.probs.size(), &st);
      cudaMemcpy(r1.data(), b.dC, b.nC * 8, cudaMemcpyDeviceToHost);
      fill_const_device(b.dC, b.nC, 0.0);
      int rc2 = ggemm_grouped_dgemm(h, p, &c, b.probs.data(), (int)b.probs.size(), &st);
      cudaMemcpy(r2.data(), b.dC, b.nC * 8, cudaMemcpyDeviceToHost);
      CHECK(rc1 == 0 && rc2 == 0 && memcmp(r1.data(), r2.data(), b.nC * 8) == 0,
            "%s not bitwise reproducible on %s", ggemm_path_name(p), w.name.c_str());
      ggemm_destroy(h);
    }
    free_workload(&b);
  }
}

int main(int argc, char** argv) {
  std::string set = "test", paths_s = "all", phis_s = "0,0.5,1,2", csv;
  bool edges = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--set") set = nxt();
    else if (a == "--paths") paths_s = nxt();
    else if (a == "--phis") phis_s = nxt();
    else if (a == "--csv") csv = nxt();
    else if (a == "--no-edges") edges = false;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  // num_repaired is only measured when GGEMM_STATS_REPAIR=1 (syncs the stream; fine outside timing)
  setenv("GGEMM_STATS_REPAIR", "1", 0);
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { fprintf(stderr, "no usable CUDA device\n"); return 1; }
  printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  if (!csv.empty()) {
    g_csv = fopen(csv.c_str(), "w");
    if (g_csv) fprintf(g_csv, "case,path,G,err_cw,err_over_gamma,err_vs_native,rel_max,nonfinite,"
                              "num_fallback,launches,moduli_min,moduli_max,status,n_over_gamma,num_uncertified,"
                              "num_repaired,gate\n");
  }
  cublasHandle_t cub; cublasCreate(&cub);
  auto paths = parse_paths(paths_s);
  if (edges) test_edges(paths, cub);
  test_workloads(set, paths, parse_doubles(phis_s), cub);
  cublasDestroy(cub);
  if (g_csv) fclose(g_csv);
  printf("\n==== %d passed, %d failed ====\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
