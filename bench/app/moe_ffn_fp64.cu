// bench/app/moe_ffn_fp64.cu -- application-level benchmark: FP64 Mixture-of-Experts FFN forward
// (owner: agent-E). All GEMMs go through ggemm_grouped_dgemm(); no other GEMM library is called.
//
// ---------------------------------------------------------------------------------------------
// Why an FP64 MoE is a reasonable scenario (and where it is not)
//  * NOT for LLM inference/training: LLM MoE layers run in BF16/FP8; FP64 there has no
//    justification and we do not claim one.
//  * Scientific ML is where FP64 MoE-shaped compute appears:
//    - Machine-learned interatomic potentials of the Behler-Parrinello / DeePMD family evaluate
//      a separate per-element fitting network for every atom: each atom is hard-routed to the
//      network of its chemical element (= top-1 MoE with gate 1, run with --topk 1). Expert loads
//      follow the composition and are strongly skewed (e.g. H/C/O dominate, dopants are rare),
//      which is what the Zipf routing below models. These codes are commonly run with float64
//      networks inside molecular dynamics, because force/energy round-off accumulates over
//      millions of time steps (energy drift in NVE runs).
//    - MoE surrogates / neural operators that are coupled into an FP64 PDE solver
//      (e.g. a learned closure called from inside a Newton or time-stepping loop): the
//      surrogate is evaluated in the solver's precision so that it does not become the
//      accuracy floor of the solver.
//  * In both cases the GEMM shapes are exactly a grouped GEMM with data-dependent, skewed m_e,
//    so this is the case the grouped FP64-emulation is designed for.
//  * Simplification (stated honestly): the router is a fixed-seed synthetic Zipf router
//    (no gate GEMM, no softmax over learned logits); its cost is included in the "route" stage.
//
// ---------------------------------------------------------------------------------------------
// Pipeline (column-major everywhere, T tokens, d = d_model, f = d_ff, E experts, top-k):
//   route   : per token, k distinct experts drawn from Zipf(s) over E (counter-based hash of
//             (seed, token, slot) -> deterministic), gate weights normalised to sum 1;
//             deterministic per-expert rank (block scan), counts D2H (pinned), host builds the
//             grouped descriptors (m_e = count_e), expert offsets H2D (pinned).  This host
//             round trip is what a real MoE layer also does and is INCLUDED in the timing.
//   gather  : Xg_e (m_e x d) <- rows of X (T x d) routed to e
//   up      : grouped GEMM  H_e  (m_e x f) = Xg_e (m_e x d) * W1_e (d x f)        [ggemm]
//   act     : Ha = act(H)   (tanh default, or tanh-GELU)
//   down    : grouped GEMM  Yd_e (m_e x d) = Ha_e (m_e x f) * W2_e (f x d)        [ggemm]
//   scatter : Y[t,:] = sum_j g_tj * Yd_{e_tj}[slot_tj, :]   (fixed order j -> deterministic)
// Total time and every stage are timed with CUDA events on one stream; stage % = mean stage
// time / mean total time. All buffers, handles, events and pinned host memory are allocated
// before the timed region; the timed region does no allocation.
//
// Correctness: every path is compared against the same pipeline run with NATIVE_LOOP
// (cublasDgemm per expert, native FP64). Routing is deterministic, so buffer layouts match.
//   err_up_cw = max |H - H_ref| / (|Xg| |W1|)                          (pure GEMM, same inputs)
//   err_y_cw  = max |Y - Y_ref| / den_y,
//               den_y = sum_j |g_tj| ( |W2|^T (|Ha_ref| + L |Xg||W1|) )_row   (L = sup|act'|)
//               i.e. the first-order componentwise sensitivity of the whole layer to
//               perturbations of size |.| in both GEMMs (classic Higham-style bound shape).
//   *_over_bound = err_cw / (2 gamma_n): both the path and the reference satisfy the FP64
//               a-priori bound gamma_n * den, so their difference is <= 2 gamma_n den;
//               n = d for the up GEMM, n = max(d,f) + k + 4 for the layer (k-term from the
//               scatter sum, +4 for the activation's few-ulp evaluation error). <= 1 means
//               "FP64-level"; the value is a diagnostic, not a proof.
//
// Usage: moe_ffn_fp64 [--tokens 4096] [--experts 16] [--topk 2] [--d-model 1024] [--d-ff 2048]
//                     [--zipf 1.0] [--phi 0.5] [--act tanh|gelu] [--seed 1234] [--moduli 0]
//                     [--paths all] [--reps 10] [--warmup 3] [--csv moe_ffn_fp64.csv] [--append]
// ---------------------------------------------------------------------------------------------
#include <cub/block/block_scan.cuh>
#include "app_common.h"

__device__ __forceinline__ uint64_t mix64d(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ double hash_u01(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t z = mix64d(seed + 0x9E3779B97F4A7C15ull * (a * 0x100000001B3ull + b * 0x1F + c + 1));
  return (z >> 11) * 0x1.0p-53;
}

// ---- route: Zipf top-k without replacement + gates
__global__ void k_route(int T, int E, int topk, const double* cdf, uint64_t seed, int* expert, double* gate) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int chosen[8]; double w[8]; double ws = 0;
  for (int j = 0; j < topk; ++j) {
    int e = 0;
    for (int a = 0; a < 1024; ++a) {
      double u = hash_u01(seed, t, j, a);
      int lo = 0, hi = E - 1;                     // first e with cdf[e] > u
      while (lo < hi) { int mid = (lo + hi) / 2; if (cdf[mid] > u) hi = mid; else lo = mid + 1; }
      e = lo;
      bool dup = false;
      for (int q = 0; q < j; ++q) dup |= (chosen[q] == e);
      if (!dup) break;
    }
    chosen[j] = e;
    w[j] = 0.05 + hash_u01(seed ^ 0x5A5A5A5Aull, t, j, 0);
    ws += w[j];
  }
  for (int j = 0; j < topk; ++j) { expert[t * topk + j] = chosen[j]; gate[t * topk + j] = w[j] / ws; }
}

// ---- deterministic rank of every assignment inside its expert (one block per expert)
constexpr int RANK_THREADS = 256;
__global__ void k_rank(int A, const int* expert, int* slot, int* counts) {
  using Scan = cub::BlockScan<int, RANK_THREADS>;
  __shared__ typename Scan::TempStorage tmp;
  int e = blockIdx.x, running = 0;
  for (int base = 0; base < A; base += RANK_THREADS) {
    int i = base + threadIdx.x;
    int flag = (i < A && expert[i] == e) ? 1 : 0, pre, agg;
    Scan(tmp).ExclusiveSum(flag, pre, agg);
    if (flag) slot[i] = running + pre;
    running += agg;
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[e] = running;
}

// ---- gather: Xg (expert-major, each expert col-major m_e x d, ld = m_e)
__global__ void k_gather(int T, int d, int topk, const double* X, const int* expert, const int* slot,
                         const int* off, const int* counts, double* Xg) {
  size_t A = (size_t)T * topk, n = A * d;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
    size_t i = idx % A, c = idx / A;
    int t = (int)(i / topk), e = expert[i];
    Xg[(size_t)off[e] * d + slot[i] + c * (size_t)counts[e]] = X[t + c * (size_t)T];
  }
}

// ---- activation (also used to build |act| for the error denominator)
__device__ __forceinline__ double act_f(double x, int kind) {
  if (kind == 0) return tanh(x);
  const double c = 0.7978845608028654;   // sqrt(2/pi)
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}
__global__ void k_act(const double* H, double* Ha, size_t n, int kind) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    Ha[i] = act_f(H[i], kind);
}

// ---- scatter: Y (T x d col-major) = sum_j g_tj * Yd_e[slot, :]
__global__ void k_scatter(int T, int d, int topk, const double* Yd, const int* expert, const int* slot,
                          const double* gate, const int* off, const int* counts, double* Y, int abs_gate) {
  size_t n = (size_t)T * d;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
    int t = (int)(idx % T); size_t c = idx / T;
    double acc = 0;
    for (int j = 0; j < topk; ++j) {
      int i = t * topk + j, e = expert[i];
      double g = abs_gate ? fabs(gate[i]) : gate[i];
      acc += g * Yd[(size_t)off[e] * d + slot[i] + c * (size_t)counts[e]];
    }
    Y[idx] = acc;
  }
}
// Q = |Ha| + L * P   (denominator helper)
__global__ void k_den_mix(const double* Ha, const double* P, double L, double* Q, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    Q[i] = fabs(Ha[i]) + L * P[i];
}

struct MoE {
  int T, E, topk, d, f, act;
  size_t A;                                     // T * topk assignments
  // device
  double *X, *W1, *W2, *Xg, *H, *Ha, *Yd, *Y, *gate, *cdf;
  int *expert, *slot, *counts, *off;
  // pinned host
  int *h_counts, *h_off;
  std::vector<ggemm_problem_t> up, down;        // capacity E (reserved; never reallocates)
};

// Builds the grouped descriptors from h_counts (host; part of the timed "route" stage).
static void build_desc(MoE& m, const double* Xg, const double* W1, double* H, const double* Ha,
                       const double* W2, double* Yd) {
  m.up.clear(); m.down.clear();
  int o = 0;
  for (int e = 0; e < m.E; ++e) {
    m.h_off[e] = o;
    int me = m.h_counts[e];
    if (me > 0) {
      int ld = me;
      ggemm_problem_t u{me, m.f, m.d, Xg + (size_t)o * m.d, ld, W1 + (size_t)e * m.d * m.f, m.d,
                        H + (size_t)o * m.f, ld, 1.0, 0.0};
      ggemm_problem_t w{me, m.d, m.f, Ha + (size_t)o * m.f, ld, W2 + (size_t)e * m.f * m.d, m.f,
                        Yd + (size_t)o * m.d, ld, 1.0, 0.0};
      m.up.push_back(u); m.down.push_back(w);
    }
    o += me;
  }
}

int main(int argc, char** argv) {
  int T = 4096, E = 16, topk = 2, d = 1024, f = 2048, reps = 10, warmup = 3, nmod = 0;
  double zipf = 1.0, phi = 0.5; uint64_t seed = 1234;
  std::string paths_s = "all", csv = "moe_ffn_fp64.csv", act_s = "tanh";
  bool append = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--tokens") T = atoi(nxt().c_str());
    else if (a == "--experts") E = atoi(nxt().c_str());
    else if (a == "--topk") topk = atoi(nxt().c_str());
    else if (a == "--d-model") d = atoi(nxt().c_str());
    else if (a == "--d-ff") f = atoi(nxt().c_str());
    else if (a == "--zipf") zipf = atof(nxt().c_str());
    else if (a == "--phi") phi = atof(nxt().c_str());
    else if (a == "--act") act_s = nxt();
    else if (a == "--seed") seed = strtoull(nxt().c_str(), nullptr, 10);
    else if (a == "--moduli") nmod = atoi(nxt().c_str());
    else if (a == "--paths") paths_s = nxt();
    else if (a == "--reps") reps = atoi(nxt().c_str());
    else if (a == "--warmup") warmup = atoi(nxt().c_str());
    else if (a == "--csv") csv = nxt();
    else if (a == "--append") append = true;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (T <= 0 || d <= 0 || f <= 0 || E <= 0) { fprintf(stderr, "--tokens/--d-model/--d-ff/--experts must be > 0\n"); return 2; }
  if (topk < 1 || topk > 8 || topk > E) { fprintf(stderr, "need 1 <= topk <= min(8,E)\n"); return 2; }
  const int act = (act_s == "gelu") ? 1 : 0;
  const double L = act ? 1.13 : 1.0;            // sup |act'|: tanh 1, tanh-GELU ~1.129
  auto paths = app_parse_paths(paths_s);
  const std::string envs = app_env_str();
  std::string gpu = app_gpu_name();

  MoE m{}; m.T = T; m.E = E; m.topk = topk; m.d = d; m.f = f; m.act = act; m.A = (size_t)T * topk;
  const size_t A = m.A;
  cudaStream_t s; APP_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // ------------------------------------------------ allocation (outside every timed region)
  auto dalloc = [](size_t n) { double* p; APP_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(double))); return p; };
  m.X = dalloc((size_t)T * d); m.W1 = dalloc((size_t)E * d * f); m.W2 = dalloc((size_t)E * f * d);
  m.Xg = dalloc(A * d); m.H = dalloc(A * f); m.Ha = dalloc(A * f); m.Yd = dalloc(A * d); m.Y = dalloc((size_t)T * d);
  m.gate = dalloc(A); m.cdf = dalloc(E);
  APP_CUDA(cudaMalloc(&m.expert, A * sizeof(int))); APP_CUDA(cudaMalloc(&m.slot, A * sizeof(int)));
  APP_CUDA(cudaMalloc(&m.counts, E * sizeof(int))); APP_CUDA(cudaMalloc(&m.off, E * sizeof(int)));
  APP_CUDA(cudaMallocHost(&m.h_counts, E * sizeof(int))); APP_CUDA(cudaMallocHost(&m.h_off, E * sizeof(int)));
  memset(m.h_counts, 0, E * sizeof(int)); memset(m.h_off, 0, E * sizeof(int));
  m.up.reserve(E); m.down.reserve(E);
  // reference / error-denominator buffers
  double *H_ref = dalloc(A * f), *Ha_ref = dalloc(A * f), *Y_ref = dalloc((size_t)T * d);
  double *aXg = dalloc(A * d), *aW1 = dalloc((size_t)E * d * f), *aW2 = dalloc((size_t)E * f * d);
  double *P = dalloc(A * f), *Q = dalloc(A * f), *R = dalloc(A * d), *den_y = dalloc((size_t)T * d);
  unsigned long long* d_u; double* d_s;
  APP_CUDA(cudaMalloc(&d_u, 2 * sizeof(unsigned long long))); APP_CUDA(cudaMalloc(&d_s, 2 * sizeof(double)));
  const int NEV = 7;
  cudaEvent_t ev[NEV]; for (auto& e : ev) APP_CUDA(cudaEventCreate(&e));

  // ------------------------------------------------ deterministic data
  fill_ozaki_device(m.X, (size_t)T * d, phi, seed ^ 0x11, s);
  fill_ozaki_device(m.W1, (size_t)E * d * f, phi, seed ^ 0x22, s);
  fill_ozaki_device(m.W2, (size_t)E * f * d, phi, seed ^ 0x33, s);
  app_scale(m.W1, (size_t)E * d * f, 2.0 / std::sqrt((double)d), s);   // keep pre-activations O(1)
  app_scale(m.W2, (size_t)E * f * d, 2.0 / std::sqrt((double)f), s);
  {
    std::vector<double> c(E); double z = 0;
    for (int e = 0; e < E; ++e) z += std::pow(e + 1.0, -zipf);
    double acc = 0;
    for (int e = 0; e < E; ++e) { acc += std::pow(e + 1.0, -zipf) / z; c[e] = acc; }
    c[E - 1] = 1.0 + 1e-12;
    APP_CUDA(cudaMemcpyAsync(m.cdf, c.data(), E * sizeof(double), cudaMemcpyHostToDevice, s));
    APP_CUDA(cudaStreamSynchronize(s));
  }

  // handle sizing: worst case over both GEMMs; independent of the routing outcome
  size_t maxA = A * (size_t)std::max(d, f) + 1, maxB = (size_t)E * d * f + 1, maxC = A * (size_t)std::max(d, f) + 1;

  // One forward pass. ev != nullptr -> record stage events. Returns ggemm rc (0 = ok).
  auto forward = [&](ggemm_handle_t h, ggemm_path_t p, const ggemm_config_t& cfg, cudaEvent_t* e,
                     ggemm_stats_t* su, ggemm_stats_t* sd) -> int {
    if (e) APP_CUDA(cudaEventRecord(e[0], s));
    k_route<<<(T + 127) / 128, 128, 0, s>>>(T, E, topk, m.cdf, seed, m.expert, m.gate);
    k_rank<<<E, RANK_THREADS, 0, s>>>((int)A, m.expert, m.slot, m.counts);
    APP_CUDA(cudaMemcpyAsync(m.h_counts, m.counts, E * sizeof(int), cudaMemcpyDeviceToHost, s));
    APP_CUDA(cudaStreamSynchronize(s));
    build_desc(m, m.Xg, m.W1, m.H, m.Ha, m.W2, m.Yd);
    APP_CUDA(cudaMemcpyAsync(m.off, m.h_off, E * sizeof(int), cudaMemcpyHostToDevice, s));
    if (e) APP_CUDA(cudaEventRecord(e[1], s));
    k_gather<<<app_grid(A * d), 256, 0, s>>>(T, d, topk, m.X, m.expert, m.slot, m.off, m.counts, m.Xg);
    if (e) APP_CUDA(cudaEventRecord(e[2], s));
    int rc = ggemm_grouped_dgemm(h, p, &cfg, m.up.data(), (int)m.up.size(), su);
    if (e) APP_CUDA(cudaEventRecord(e[3], s));
    k_act<<<app_grid(A * f), 256, 0, s>>>(m.H, m.Ha, A * f, act);
    if (e) APP_CUDA(cudaEventRecord(e[4], s));
    rc |= ggemm_grouped_dgemm(h, p, &cfg, m.down.data(), (int)m.down.size(), sd);
    if (e) APP_CUDA(cudaEventRecord(e[5], s));
    k_scatter<<<app_grid((size_t)T * d), 256, 0, s>>>(T, d, topk, m.Yd, m.expert, m.slot, m.gate, m.off,
                                                      m.counts, m.Y, 0);
    if (e) APP_CUDA(cudaEventRecord(e[6], s));
    return rc;
  };

  // ------------------------------------------------ reference (NATIVE_LOOP) + error denominators
  int ref_ok = 0, load_max = 0, load_min = 0, n_active = 0;
  {
    ggemm_handle_t h = nullptr;
    if (ggemm_create(&h, E, maxA, maxB, maxC, s) == 0 && h) {
      ggemm_config_t cfg = app_cfg(GGEMM_PATH_NATIVE_LOOP, 0);
      ggemm_stats_t st;
      if (forward(h, GGEMM_PATH_NATIVE_LOOP, cfg, nullptr, &st, &st) == 0) {
        APP_CUDA(cudaMemcpyAsync(H_ref, m.H, A * f * sizeof(double), cudaMemcpyDeviceToDevice, s));
        APP_CUDA(cudaMemcpyAsync(Ha_ref, m.Ha, A * f * sizeof(double), cudaMemcpyDeviceToDevice, s));
        APP_CUDA(cudaMemcpyAsync(Y_ref, m.Y, (size_t)T * d * sizeof(double), cudaMemcpyDeviceToDevice, s));
        // P = |Xg||W1| ; Q = |Ha_ref| + L P ; R = Q |W2| ; den_y = scatter(|g|, R)
        app_abs(m.Xg, aXg, A * d, s); app_abs(m.W1, aW1, (size_t)E * d * f, s); app_abs(m.W2, aW2, (size_t)E * f * d, s);
        build_desc(m, aXg, aW1, P, Q, aW2, R);
        int rc = ggemm_grouped_dgemm(h, GGEMM_PATH_NATIVE_LOOP, &cfg, m.up.data(), (int)m.up.size(), &st);
        k_den_mix<<<app_grid(A * f), 256, 0, s>>>(Ha_ref, P, L, Q, A * f);
        rc |= ggemm_grouped_dgemm(h, GGEMM_PATH_NATIVE_LOOP, &cfg, m.down.data(), (int)m.down.size(), &st);
        k_scatter<<<app_grid((size_t)T * d), 256, 0, s>>>(T, d, topk, R, m.expert, m.slot, m.gate, m.off,
                                                          m.counts, den_y, 1);
        APP_CUDA(cudaStreamSynchronize(s));
        ref_ok = (rc == 0);
      }
      ggemm_destroy(h);
    }
    if (ref_ok) {   // h_counts is only valid after a successful routing pass
      load_max = *std::max_element(m.h_counts, m.h_counts + E);
      load_min = *std::min_element(m.h_counts, m.h_counts + E);
      for (int e = 0; e < E; ++e) n_active += m.h_counts[e] > 0;
    }
  }
  if (!ref_ok) fprintf(stderr, "WARNING: NATIVE_LOOP reference failed; err columns will be NaN\n");
  const double gemm_flops = 2.0 * (double)A * d * f * 2.0;
  const double bnd_up = 2.0 * app_gamma(d), bnd_y = 2.0 * app_gamma((long long)std::max(d, f) + topk + 4);

  FILE* fo = fopen(csv.c_str(), append ? "a" : "w");
  if (!fo) { perror(csv.c_str()); return 1; }
  fseek(fo, 0, SEEK_END);
  if (ftell(fo) == 0)
    fprintf(fo, "app,path,T,E,topk,d_model,d_ff,zipf,phi,act,seed,num_moduli_cfg,experts_active,load_max,load_min,"
                "gemm_flops,total_ms_median,total_ms_mean,wall_ms_median,route_ms,gather_ms,up_ms,act_ms,down_ms,scatter_ms,"
                "route_pct,gather_pct,up_pct,act_pct,down_pct,scatter_pct,gemm_pct,eff_GFLOPS,"
                "launches_up,launches_down,fallback_up,fallback_down,moduli_min,moduli_max,"
                "err_up_cw,err_up_over_bound,err_y_cw,err_y_over_bound,rel_fro_y,nonfinite,reps,gpu,env,status\n");
  fprintf(stderr, "device: %s  T=%d E=%d topk=%d d=%d f=%d zipf=%.2f  loads max/min=%d/%d active=%d\n",
          gpu.c_str(), T, E, topk, d, f, zipf, load_max, load_min, n_active);

  const char* stage[6] = {"route", "gather", "up", "act", "down", "scatter"};
  for (auto p : paths) {
    ggemm_config_t cfg = app_cfg(p, nmod);
    ggemm_handle_t h = nullptr;
    const char* status = "ok";
    if (ggemm_create(&h, E, maxA, maxB, maxC, s) != 0 || !h) status = "create_failed";
    ggemm_stats_t su{}, sd{};
    int rc = 0;
    for (int r = 0; h && rc == 0 && r < warmup; ++r) rc = forward(h, p, cfg, nullptr, &su, &sd);
    if (h) APP_CUDA(cudaStreamSynchronize(s));
    std::vector<double> tot, wall, st[6];
    for (int r = 0; h && rc == 0 && r < reps; ++r) {
      auto w0 = std::chrono::steady_clock::now();
      rc = forward(h, p, cfg, ev, &su, &sd);
      APP_CUDA(cudaEventSynchronize(ev[6]));
      auto w1 = std::chrono::steady_clock::now();
      if (rc != 0) break;   // failed iteration is not timed
      float ms;
      APP_CUDA(cudaEventElapsedTime(&ms, ev[0], ev[6])); tot.push_back(ms);
      for (int k = 0; k < 6; ++k) { APP_CUDA(cudaEventElapsedTime(&ms, ev[k], ev[k + 1])); st[k].push_back(ms); }
      wall.push_back(std::chrono::duration<double, std::milli>(w1 - w0).count());
    }
    if (h && rc != 0) status = "call_failed";
    if (cudaGetLastError() != cudaSuccess) status = "cuda_error";
    AppErr eu, ey;
    if (h && rc == 0 && ref_ok) {   // untimed checking pass on poisoned outputs
      fill_const_device(m.H, A * f, NAN, s); fill_const_device(m.Yd, A * d, NAN, s);
      fill_const_device(m.Y, (size_t)T * d, NAN, s);
      // P (= |Xg||W1|) is the up-GEMM denominator
      if (forward(h, p, cfg, nullptr, &su, &sd) == 0) {
        eu = app_compare(m.H, H_ref, P, A * f, d_u, d_s, s);
        ey = app_compare(m.Y, Y_ref, den_y, (size_t)T * d, d_u, d_s, s);
      } else status = "check_failed";
    }
    double tmean = app_mean(tot), sm[6];
    for (int k = 0; k < 6; ++k) sm[k] = app_mean(st[k]);
    double tmed = app_median(tot);
    fprintf(fo, "moe_ffn,%s,%d,%d,%d,%d,%d,%.3f,%.2f,%s,%llu,%d,%d,%d,%d,%.6e,%.6f,%.6f,%.6f",
            app_path_name(p), T, E, topk, d, f, zipf, phi, act_s.c_str(), (unsigned long long)seed, nmod,
            n_active, load_max, load_min, gemm_flops, tmed, tmean, app_median(wall));
    for (int k = 0; k < 6; ++k) fprintf(fo, ",%.6f", sm[k]);
    for (int k = 0; k < 6; ++k) fprintf(fo, ",%.3f", 100.0 * sm[k] / tmean);
    fprintf(fo, ",%.3f,%.3f,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%llu,%d,%s,%s,%s\n",
            100.0 * (sm[2] + sm[4]) / tmean, gemm_flops / (tmed * 1e-3) / 1e9,
            su.kernel_launches, sd.kernel_launches, su.num_fallback, sd.num_fallback,
            std::min(su.moduli_min, sd.moduli_min), std::max(su.moduli_max, sd.moduli_max),
            eu.cw, eu.cw / bnd_up, ey.cw, ey.cw / bnd_y, ey.rel_fro, eu.bad + ey.bad, (int)tot.size(),
            gpu.c_str(), envs.c_str(), status);
    fflush(fo);
    if (!strcmp(status, "cuda_error")) { fprintf(stderr, "CUDA error (sticky) -> abort\n"); fclose(fo); exit(1); }
    fprintf(stderr, "%-15s total %9.3f ms (%8.1f GFLOPS)", app_path_name(p), tmed, gemm_flops / (tmed * 1e-3) / 1e9);
    for (int k = 0; k < 6; ++k) fprintf(stderr, " %s %4.1f%%", stage[k], 100.0 * sm[k] / tmean);
    fprintf(stderr, "  err_up=%.2e err_y=%.2e (x%.2g bound) %s\n", eu.cw, ey.cw, ey.cw / bnd_y, status);
    if (h) ggemm_destroy(h);
  }
  fclose(fo);
  return 0;
}
