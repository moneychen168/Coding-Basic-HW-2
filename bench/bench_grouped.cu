// bench/bench_grouped.cu -- end-to-end timing of every ggemm path (owner: agent-B).
// Depends only on include/ggemm.h (+ bench/workloads, tests/reference for err columns).
//
// Methodology (identical for native / starter / ours, so the comparison is like for like):
//  * Same workload, same data (fixed seed), same handle sizing, same stream for all paths.
//  * Everything that allocates (inputs, reference, ggemm_create workspace, events) happens
//    before the timed region; the timed region is exactly ONE ggemm_grouped_dgemm() call.
//  * Time = cudaEvent elapsed between an event recorded before the call and one recorded
//    after it returns, on the handle's stream. Host-side planning, descriptor packing and
//    the descriptor H2D copy are therefore INCLUDED (end-to-end), as is any host sync the
//    library does internally. Host wall time of call+sync is reported too (wall_ms).
//  * W warm-up calls (JIT/lazy module loading, cuBLAS heuristics, clocks), then R timed
//    calls; time_ms = median, time_ms_min also reported. beta=0, so repeated calls are
//    idempotent and no reset is needed between reps. L2 is not flushed (steady state of a
//    repeatedly called kernel; most MoE/hetero operand sets exceed the 50-96 MB L2 anyway).
//  * effective_GFLOPS = sum_g 2 m_g n_g k_g / time  (FP64-equivalent: the emulated paths
//    get credit only for the FP64 flops they replace, not for INT8 ops executed).
//  * err columns come from one extra (untimed) call compared against the double-double
//    reference; err_vs_native divides by cuBLAS DGEMM's error on the same data.
//
// Usage: bench_grouped [--set bench|quick|test] [--paths all|a,b] [--phis 0.5]
//                      [--moduli 0] [--reps 20] [--warmup 5] [--csv out.csv] [--no-check]
//                      [--only substr] [--moe-counts file:n:k] [--dump-shapes f.csv [--dump-only]]
//   [--repair-modes default,off,count,on] [--count-call] [--plan-csv plan.csv]
//     --count-call: counters (num_uncertified/num_repaired) come from one extra UNTIMED call on a
//       fresh handle created with GGEMM_STATS_REPAIR=1; timed calls never read counters.
//     rho_budget column = env GGEMM_RHO_BUDGET (read once per process by the planner).
//   paths "all" = native_loop,native_grouped,emu_loop,emu_grouped,auto,emu_grouped_reuse,emu_loop_batched
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "ggemm.h"
// NVTX range "bench:timed" around each timed call (e0..e1). -DGGEMM_NVTX=0 compiles it out.
#ifndef GGEMM_NVTX
#define GGEMM_NVTX 1
#endif
#if GGEMM_NVTX
#include <nvtx3/nvToolsExt.h>
#define BENCH_NVTX_PUSH(name) nvtxRangePushA(name)
#define BENCH_NVTX_POP() nvtxRangePop()
#else
#define BENCH_NVTX_PUSH(name) do {} while (0)
#define BENCH_NVTX_POP() do {} while (0)
#endif
#include "workloads.h"
#include "../tests/reference.h"

static bool is_emu(ggemm_path_t p) {
  return ggemm_path_is_emu(p);   // incl. emu_grouped_reuse, emu_loop_batched
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n == 0 ? NAN : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

// ---- repair mode (ggemm_config_t.repair_mode, include/ggemm.h) -------------------------
#if defined(GGEMM_REPAIR_ON) && defined(GGEMM_REPAIR_COUNT)
#define BENCH_HAS_REPAIR 1
static void set_repair_mode(ggemm_config_t* c, int m) { c->repair_mode = m; }
static const char* repair_mode_name(int m) {
  switch (m) { case GGEMM_REPAIR_DEFAULT: return "default"; case GGEMM_REPAIR_OFF: return "off";
               case GGEMM_REPAIR_COUNT: return "count"; case GGEMM_REPAIR_ON: return "on"; }
  return "?";
}
static int repair_mode_from_name(const std::string& s) {
  if (s == "default") return GGEMM_REPAIR_DEFAULT;
  if (s == "off") return GGEMM_REPAIR_OFF;
  if (s == "count") return GGEMM_REPAIR_COUNT;
  if (s == "on") return GGEMM_REPAIR_ON;
  return -1;
}
#else   // header without repair modes: only "default" is accepted
static void set_repair_mode(ggemm_config_t*, int) {}
static const char* repair_mode_name(int) { return "default"; }
static int repair_mode_from_name(const std::string& s) { return s == "default" ? 0 : -1; }
#endif
static std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> v; std::string cur;
  for (char c : s) { if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) v.push_back(cur);
  return v;
}

// ---- planner decisions (N distribution) via the internal planner API (src/planner.h) -----
#if __has_include("planner.h")
#include "planner.h"
static void dump_plan(FILE* fp, const Workload& w, const std::vector<ggemm_path_t>& paths,
                      const std::string& rho, const cudaDeviceProp& prop) {
  ggemm_device_model_t dm;
  ggemm_device_model_init(&dm, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  std::vector<ggemm_problem_t> pr;
  for (auto& s : w.shapes) {   // planner only reads shapes/scalars; pointers are never touched
    ggemm_problem_t p; memset(&p, 0, sizeof p);
    p.m = s.m; p.n = s.n; p.k = s.k; p.lda = std::max(1, s.m); p.ldb = std::max(1, s.k); p.ldc = std::max(1, s.m);
    p.alpha = 1.0; p.beta = 0.0; pr.push_back(p);
  }
  for (auto path : paths) {
    if (path != GGEMM_PATH_AUTO && path != GGEMM_PATH_EMU_GROUPED) continue;
    ggemm_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.target_rel_err = 1e-15; cfg.allow_fallback = (path == GGEMM_PATH_AUTO);
    std::vector<ggemm_plan_entry_t> out(pr.size());
    if (ggemm_plan(pr.data(), (int)pr.size(), &cfg, path, &dm, out.data()) != 0) continue;
    std::map<std::pair<int, int>, std::pair<long long, long long>> hist;   // (route,N) -> (problems, entries)
    for (size_t i = 0; i < pr.size(); ++i) {
      auto& c = hist[{out[i].route, out[i].route == GGEMM_ROUTE_EMU ? out[i].num_moduli : 0}];
      c.first++; c.second += (long long)pr[i].m * pr[i].n;
    }
    for (auto& kv : hist)
      fprintf(fp, "%s,%s,%s,%d,%d,%lld,%lld\n", w.name.c_str(), ggemm_path_name(path), rho.c_str(),
              kv.first.first, kv.first.second, kv.second.first, kv.second.second);
  }
  fflush(fp);
}
#else
static void dump_plan(FILE*, const Workload&, const std::vector<ggemm_path_t>&, const std::string&,
                      const cudaDeviceProp&) { fprintf(stderr, "--plan-csv: planner.h not available\n"); }
#endif

int main(int argc, char** argv) {
  std::string set = "bench", paths_s = "all", phis_s = "0.5", moduli_s = "0", csv = "bench.csv", only, moe_counts;
  int reps = 20, warmup = 5;
  bool check = true, dump_only = false;
  std::string dump_shapes, rmodes_s = "default", plan_csv;
  bool count_call = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--set") set = nxt();
    else if (a == "--paths") paths_s = nxt();
    else if (a == "--phis") phis_s = nxt();
    else if (a == "--moduli") moduli_s = nxt();
    else if (a == "--reps") reps = atoi(nxt().c_str());
    else if (a == "--warmup") warmup = atoi(nxt().c_str());
    else if (a == "--csv") csv = nxt();
    else if (a == "--only") only = nxt();
    else if (a == "--moe-counts") moe_counts = nxt();
    else if (a == "--no-check") check = false;
    else if (a == "--dump-shapes") dump_shapes = nxt();
    else if (a == "--dump-only") dump_only = true;
    else if (a == "--repair-modes") rmodes_s = nxt();
    else if (a == "--count-call") count_call = true;
    else if (a == "--plan-csv") plan_csv = nxt();
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { fprintf(stderr, "no usable CUDA device\n"); return 1; }
  std::string gpu = prop.name;
  for (auto& c : gpu) if (c == ',') c = ' ';
  fprintf(stderr, "device: %s (sm_%d%d), reps=%d warmup=%d\n", prop.name, prop.major, prop.minor, reps, warmup);

  auto wls = make_workloads(set);
  if (!moe_counts.empty()) {   // file:n:k  (real routing trace)
    size_t c1 = moe_counts.find(':'), c2 = moe_counts.rfind(':');
    if (c1 != std::string::npos && c2 > c1) {
      Workload w = make_moe_from_counts(moe_counts.substr(0, c1), atoi(moe_counts.substr(c1 + 1, c2 - c1 - 1).c_str()),
                                        atoi(moe_counts.substr(c2 + 1).c_str()), "moe_trace");
      if (!w.shapes.empty()) wls.push_back(w);
    }
  }
  // --dump-shapes FILE: workload,m,n,k (one row per problem) for scripts/analysis/roofline.py --shapes
  if (!dump_shapes.empty()) {
    FILE* fs = fopen(dump_shapes.c_str(), "w");
    if (!fs) { perror(dump_shapes.c_str()); return 1; }
    fprintf(fs, "workload,m,n,k\n");
    for (auto& w : wls) {
      if (!only.empty() && w.name.find(only) == std::string::npos) continue;
      for (auto& sh : w.shapes) fprintf(fs, "%s,%d,%d,%d\n", w.name.c_str(), sh.m, sh.n, sh.k);
    }
    fclose(fs);
    fprintf(stderr, "shapes -> %s\n", dump_shapes.c_str());
    if (dump_only) return 0;
  }
  auto paths = parse_paths(paths_s);
  auto phis = parse_doubles(phis_s);
  auto moduli = parse_ints(moduli_s);
  std::vector<int> rmodes;
  for (auto& r : split_list(rmodes_s)) {
    int m = repair_mode_from_name(r);
    if (m < 0) { fprintf(stderr, "repair mode '%s' unknown or not supported by include/ggemm.h\n", r.c_str()); return 2; }
    rmodes.push_back(m);
  }
  const char* rho_c = getenv("GGEMM_RHO_BUDGET");
  std::string rho_env = rho_c ? rho_c : "default";
  FILE* fplan = nullptr;
  if (!plan_csv.empty()) {
    fplan = fopen(plan_csv.c_str(), "w");
    if (!fplan) { perror(plan_csv.c_str()); return 1; }
    fprintf(fplan, "workload,path,rho_budget,route,num_moduli,problems,entries\n");
  }

  FILE* f = fopen(csv.c_str(), "w");
  if (!f) { perror(csv.c_str()); return 1; }
  fprintf(f, "workload,path,G,total_flops,time_ms,effective_GFLOPS,launches,err_max,num_fallback,"
             "family,phi,num_moduli_cfg,time_ms_min,wall_ms,err_over_gamma,err_vs_native,rel_max,"
             "moduli_min,moduli_max,reps,gpu,status,num_uncertified,num_repaired,n_over_gamma,repair_mode,"
             "rho_budget,total_entries,num_safety_fallback\n");

  cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cublasHandle_t cub; cublasCreate(&cub);

  for (auto& w : wls) {
    if (!only.empty() && w.name.find(only) == std::string::npos) continue;
    if (fplan) dump_plan(fplan, w, paths, rho_env, prop);
    for (double phi : phis) {
      WorkloadBuffers b;
      if (alloc_workload(w, phi, 1.0, 0.0, 0, &b)) { fprintf(stderr, "alloc failed: %s\n", w.name.c_str()); continue; }
      RefBuffers ref; ErrStats en; bool have_ref = false;
      if (check && ref_alloc(&ref, b) == 0 && ref_compute(b, &ref) == 0) {
        reset_C(&b); run_cublas_dgemm_loop(b, cub); ref_compare(b, &ref, &en); have_ref = true;
      }
      int G = (int)b.probs.size();
      for (auto p : paths) {
        std::vector<int> mods = is_emu(p) ? moduli : std::vector<int>{0};
        for (int nm : mods) for (int rm : (is_emu(p) ? rmodes : std::vector<int>{0})) {
          ggemm_config_t cfg; memset(&cfg, 0, sizeof cfg);   // appended fields (repair_mode) = 0 = default
          cfg.num_moduli = nm; cfg.target_rel_err = 1e-15;
          cfg.allow_fallback = (p == GGEMM_PATH_AUTO) ? 1 : 0;
          set_repair_mode(&cfg, rm);
          ggemm_handle_t h = nullptr;
          const char* status = "ok";
          std::vector<double> t, wall;
          ggemm_stats_t st; memset(&st, 0, sizeof st);
          ggemm_stats_t sc; memset(&sc, 0xff, sizeof sc);   // counters (untimed call); -1 = not measured
          // timed handle: created with GGEMM_STATS_REPAIR unset so the counter readback (a stream
          // sync) never lands inside the timed region
          if (count_call) unsetenv("GGEMM_STATS_REPAIR");
          if (ggemm_create(&h, std::max(G, 1), b.nA + 1, b.nB + 1, b.nC + 1, s) != 0 || !h) status = "create_failed";
          int rc = 0;
          for (int r = 0; h && rc == 0 && r < warmup; ++r) rc = ggemm_grouped_dgemm(h, p, &cfg, b.probs.data(), G, &st);
          if (h && cudaStreamSynchronize(s) != cudaSuccess) rc = -1;
          for (int r = 0; h && rc == 0 && r < reps; ++r) {
            auto w0 = std::chrono::steady_clock::now();
            BENCH_NVTX_PUSH("bench:timed");
            cudaEventRecord(e0, s);
            rc = ggemm_grouped_dgemm(h, p, &cfg, b.probs.data(), G, &st);
            cudaEventRecord(e1, s);
            BENCH_NVTX_POP();
            cudaEventSynchronize(e1);
            auto w1 = std::chrono::steady_clock::now();
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            t.push_back(ms);
            wall.push_back(std::chrono::duration<double, std::milli>(w1 - w0).count());
          }
          if (h && rc != 0) status = "call_failed";
          if (cudaGetLastError() != cudaSuccess) status = "cuda_error";
          if (count_call && h && rc == 0) {
            // separate, untimed call on a fresh handle created with GGEMM_STATS_REPAIR=1
            ggemm_destroy(h); h = nullptr;
            setenv("GGEMM_STATS_REPAIR", "1", 1);
            if (ggemm_create(&h, std::max(G, 1), b.nA + 1, b.nB + 1, b.nC + 1, s) != 0 || !h) status = "create_failed";
            unsetenv("GGEMM_STATS_REPAIR");
          }
          ErrStats e; double vs_native = NAN;
          if (h && rc == 0 && (have_ref || count_call)) {
            fill_const_device(b.dC, b.nC, NAN, s);   // make sure the checked C comes from this call
            rc = ggemm_grouped_dgemm(h, p, &cfg, b.probs.data(), G, &sc);
            cudaStreamSynchronize(s);
            if (rc != 0) status = "check_call_failed";
            if (have_ref) {
              ref_compare(b, &ref, &e);
              vs_native = en.err_cw > 0 ? e.err_cw / en.err_cw : NAN;
            }
          }
          double tm = median(t), tmin = t.empty() ? NAN : *std::min_element(t.begin(), t.end());
          double gfl = w.total_flops() / (tm * 1e-3) / 1e9;
          fprintf(f, "%s,%s,%d,%.6e,%.6f,%.3f,%d,%.6e,%d,%s,%.2f,%d,%.6f,%.6f,%.6e,%.6e,%.6e,%d,%d,%d,%s,%s,%lld,%lld,%lld,"
                     "%s,%s,%zu,%lld\n",
                  w.name.c_str(), ggemm_path_name(p), G, w.total_flops(), tm, gfl, st.kernel_launches,
                  have_ref ? e.err_cw : NAN, st.num_fallback, w.family.c_str(), phi, nm, tmin, median(wall),
                  have_ref ? e.ratio_gamma : NAN, vs_native, have_ref ? e.rel_max : NAN, st.moduli_min,
                  st.moduli_max, (int)t.size(), gpu.c_str(), status, stats_num_uncertified(sc), stats_num_repaired(sc),
                  have_ref ? (long long)e.n_over_gamma : -1LL, repair_mode_name(rm), rho_env.c_str(), w.elems_C(),
                  stats_num_safety_fallback(sc));
          fflush(f);
          fprintf(stderr, "%-34s %-17s N=%-2d rm=%-7s phi=%.1f  %9.4f ms  %9.1f GFLOPS  launches=%-6d fb=%-4d "
                          "err=%.2e (x%.2g native) uncert=%lld repaired=%lld %s\n",
                  w.name.c_str(), ggemm_path_name(p), nm, repair_mode_name(rm), phi, tm, gfl, st.kernel_launches,
                  st.num_fallback, have_ref ? e.err_cw : NAN, vs_native, stats_num_uncertified(sc),
                  stats_num_repaired(sc), status);
          if (!strcmp(status, "cuda_error")) { fprintf(stderr, "CUDA error -> abort\n"); fclose(f); exit(1); }
          if (h) ggemm_destroy(h);
        }
      }
      if (have_ref) ref_free(&ref);
      free_workload(&b);
    }
  }
  fclose(f);
  if (fplan) fclose(fplan);
  cublasDestroy(cub);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
  return 0;
}
