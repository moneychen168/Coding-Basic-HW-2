// bench/workloads.cu -- shape sets and deterministic data generation (owner: agent-B).
//
// ============================== WHY THESE SHAPES ==============================
// The point of the grouped path is to amortise per-call fixed costs (kernel launches,
// descriptor H2D, scale/residue prep, CRT reconstruction) over many problems, and to keep
// INT8 tensor cores busy when individual problems are too small or too unequal to fill the
// GPU. Each family isolates one regime where FP64 grouped GEMM shows up in practice and
// where a naive "loop over problems" implementation is known to be weak:
//
// (a) uniform batched small: G in {256,1024,4096}, m=n=k=s in {16,32,64,128,256}.
//     Classic batched-BLAS regime (FEM element matrices, block-sparse / tensor-network
//     contractions, Cholesky of many small blocks). Per-problem flops 2s^3 range from 8 K
//     (s=16, fully launch/latency bound) to 33 M (s=256, approaching compute bound), so the
//     sweep shows where emulation stops paying for its O(N) modulus GEMMs. G sweeps the
//     amount of parallel work available to a single fused launch. Cap: per-operand
//     elements G*s^2 <= 2^26 (512 MiB in FP64), which only drops G=4096,s=256 (needs 2 GiB
//     per operand plus reference buffers; too tight for a 32 GB RTX 5090 together with the
//     INT8 residue workspace of up to 20 moduli).
//
// (b) MoE-like: one GEMM per expert, m_e = tokens routed to expert e (m_e = 0 allowed), and
//     (n,k) = (ffn,hidden) for the up/gate projection or (hidden,ffn) for the down
//     projection. Routing load is skewed: expert popularity follows Zipf(s) with a seeded
//     random permutation of expert ids (so the hot expert is not always problem 0, which
//     would flatter a naive in-order scheduler). Each token picks top_k distinct experts.
//     Zipf s=0 is the balanced reference, s=1.2 matches the heavy head reported for
//     learned routers without strong load-balancing loss (no real trace is bundled; a real
//     per-expert count file can be loaded with make_moe_from_counts / --moe-counts).
//       E=8   top2  hidden 1024 ffn 3584 : Mixtral-8x7B ratios (4096/14336) scaled by 1/4
//                                          so B (8 experts) fits comfortably in FP64.
//       E=64  top6  hidden 2048 ffn 1408 : DeepSeek-V2-Lite / Qwen1.5-MoE fine-grained dims.
//       E=128 top8  hidden 2048 ffn 768  : Qwen3-30B-A3B dims (unscaled).
//     Two token regimes: "prefill" (T=2048/1024/512 tokens -> thousands of assignments,
//     unequal but mostly non-empty problems) and "decode" (T=64 -> many m_e = 0 or tiny,
//     where per-problem overhead and empty-problem handling dominate).
//
// (c) random heterogeneous: G in {64,256}, each of m,n,k drawn independently log-uniform in
//     [16,2048]. Stresses per-problem tile-config choice and load balancing (LPT): the
//     largest problem is ~10^6x the smallest, so any static even split of tiles fails.
//
// (d) multi-LoRA (Punica SGMV / S-LoRA style): G adapters, tokens per adapter Zipf(1.0)
//     over T=2048, hidden 4096. "shrink" = x(t_a x 4096) * A_a(4096 x r) -> n=r tiny;
//     "expand" = h(t_a x r) * B_a(r x 4096) -> k=r tiny. r=16 uniform, or mixed ranks
//     {8,16,32,64} per adapter (S-LoRA heterogeneous ranks). Small n defeats tile shapes
//     tuned for square GEMMs; small k makes the O(k) Ozaki prep cheap but the CRT
//     reconstruction (O(N*m*n)) dominant -- the planner should notice.
//
// Data: a_ij = (rand-0.5)*exp(phi*randn), phi in {0,0.5,1,2} (Ozaki et al. convention).
// Larger phi widens the exponent range inside each row/column, which is exactly what
// hurts per-row power-of-two scaling + truncation in Ozaki scheme II.
// ==============================================================================
#include "workloads.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>

#define WL_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

double Workload::total_flops() const {
  double f = 0; for (auto& s : shapes) f += 2.0 * s.m * (double)s.n * s.k; return f;
}
size_t Workload::elems_A() const { size_t t = 0; for (auto& s : shapes) t += (size_t)s.m * s.k; return t; }
size_t Workload::elems_B() const { size_t t = 0; for (auto& s : shapes) t += (size_t)s.k * s.n; return t; }
size_t Workload::elems_C() const { size_t t = 0; for (auto& s : shapes) t += (size_t)s.m * s.n; return t; }
int Workload::max_k() const { int k = 0; for (auto& s : shapes) k = std::max(k, s.k); return k; }

static const uint64_t kBaseSeed = 20261007ull;   // fixed; every workload derives from it
static uint64_t seed_of(const std::string& name) {      // FNV-1a of the name, mixed with base
  uint64_t h = 1469598103934665603ull;
  for (char c : name) { h ^= (unsigned char)c; h *= 1099511628211ull; }
  return h ^ kBaseSeed;
}

// ---------------------------------------------------------------- (a) uniform
static void add_uniform(std::vector<Workload>& out, int G, int s) {
  if ((double)G * s * s > (double)(1ull << 26)) return;   // memory cap, see header comment
  Workload w; w.family = "uniform";
  w.name = "uni_G" + std::to_string(G) + "_s" + std::to_string(s);
  w.seed = seed_of(w.name);
  w.shapes.assign(G, Shape{s, s, s});
  out.push_back(w);
}
static void add_uniform_mnk(std::vector<Workload>& out, int G, int m, int n, int k) {
  Workload w; w.family = "uniform";
  w.name = "uni_G" + std::to_string(G) + "_" + std::to_string(m) + "x" + std::to_string(n) +
           "x" + std::to_string(k);
  w.seed = seed_of(w.name);
  w.shapes.assign(G, Shape{m, n, k});
  out.push_back(w);
}

// ---------------------------------------------------------------- (b) MoE
static std::vector<int> moe_counts(int E, int topk, int T, double zipf_s, uint64_t seed) {
  SplitMix64 r(seed);
  std::vector<int> perm(E);
  for (int i = 0; i < E; ++i) perm[i] = i;
  for (int i = E - 1; i > 0; --i) std::swap(perm[i], perm[r.next() % (uint64_t)(i + 1)]);
  std::vector<double> cdf(E);
  double acc = 0;
  for (int i = 0; i < E; ++i) { acc += std::pow((double)(i + 1), -zipf_s); cdf[i] = acc; }
  for (auto& c : cdf) c /= acc;
  std::vector<int> cnt(E, 0);
  std::vector<int> picked;
  for (int t = 0; t < T; ++t) {
    picked.clear();
    while ((int)picked.size() < topk) {   // top-k distinct experts per token
      double u = r.uniform();
      int rank = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
      if (rank >= E) rank = E - 1;
      int e = perm[rank];
      if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
    }
    for (int e : picked) cnt[e]++;
  }
  return cnt;
}
static void add_moe(std::vector<Workload>& out, int E, int topk, int T, int hidden, int ffn,
                    double zipf_s, const char* regime) {
  char zs[16]; snprintf(zs, sizeof zs, "%.1f", zipf_s);
  std::string base = "moe_E" + std::to_string(E) + "_" + regime + "_z" + zs;
  auto cnt = moe_counts(E, topk, T, zipf_s, seed_of(base));
  for (int proj = 0; proj < 2; ++proj) {
    Workload w; w.family = "moe";
    w.name = base + (proj == 0 ? "_up" : "_down");
    w.seed = seed_of(w.name);
    for (int e = 0; e < E; ++e)
      w.shapes.push_back(proj == 0 ? Shape{cnt[e], ffn, hidden} : Shape{cnt[e], hidden, ffn});
    out.push_back(w);
  }
}

Workload make_moe_from_counts(const std::string& path, int n, int k, const std::string& name) {
  Workload w; w.family = "moe"; w.name = name; w.seed = seed_of(name);
  std::ifstream f(path);
  if (!f) return w;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line); int c;
    while (is >> c) w.shapes.push_back(Shape{c, n, k});
  }
  return w;
}

// ---------------------------------------------------------------- (c) heterogeneous
static void add_hetero(std::vector<Workload>& out, int G, int lo, int hi) {
  Workload w; w.family = "hetero";
  w.name = "het_G" + std::to_string(G) + "_" + std::to_string(lo) + "-" + std::to_string(hi);
  w.seed = seed_of(w.name);
  SplitMix64 r(w.seed);
  auto lu = [&]() {
    double x = std::exp(std::log((double)lo) + r.uniform() * (std::log((double)hi) - std::log((double)lo)));
    return std::max(lo, std::min(hi, (int)std::lround(x)));
  };
  for (int g = 0; g < G; ++g) { int m = lu(), n = lu(), k = lu(); w.shapes.push_back(Shape{m, n, k}); }
  out.push_back(w);
}

// ---------------------------------------------------------------- (d) multi-LoRA
static void add_lora(std::vector<Workload>& out, int G, int T, int hidden, int rank /*0=mixed*/) {
  std::string base = "lora_G" + std::to_string(G) + "_r" + (rank ? std::to_string(rank) : std::string("mix"));
  SplitMix64 r(seed_of(base));
  // tokens per adapter: Zipf(1.0) over adapters, every adapter gets >= 1 token
  std::vector<double> w(G); double acc = 0;
  for (int a = 0; a < G; ++a) { w[a] = 1.0 / (a + 1); acc += w[a]; }
  std::vector<int> t(G, 1); int left = T - G;
  for (int i = 0; i < left; ++i) {
    double u = r.uniform() * acc, c = 0; int a = 0;
    for (; a < G - 1; ++a) { c += w[a]; if (u < c) break; }
    t[a]++;
  }
  static const int mix[4] = {8, 16, 32, 64};
  std::vector<int> rk(G);
  for (int a = 0; a < G; ++a) rk[a] = rank ? rank : mix[r.next() % 4];
  for (int dir = 0; dir < 2; ++dir) {
    Workload wl; wl.family = "lora";
    wl.name = base + (dir == 0 ? "_shrink" : "_expand");
    wl.seed = seed_of(wl.name);
    for (int a = 0; a < G; ++a)
      wl.shapes.push_back(dir == 0 ? Shape{t[a], rk[a], hidden} : Shape{t[a], hidden, rk[a]});
    out.push_back(wl);
  }
}

std::vector<Workload> make_workloads(const std::string& set) {
  std::vector<Workload> v;
  if (set == "bench") {
    for (int G : {256, 1024, 4096})
      for (int s : {16, 32, 64, 128, 256}) add_uniform(v, G, s);
    for (double z : {0.0, 1.2}) {
      add_moe(v, 8, 2, 2048, 1024, 3584, z, "prefill");
      add_moe(v, 64, 6, 1024, 2048, 1408, z, "prefill");
      add_moe(v, 128, 8, 512, 2048, 768, z, "prefill");
      add_moe(v, 8, 2, 64, 1024, 3584, z, "decode");
      add_moe(v, 64, 6, 64, 2048, 1408, z, "decode");
      add_moe(v, 128, 8, 64, 2048, 768, z, "decode");
    }
    add_hetero(v, 64, 16, 2048);
    add_hetero(v, 256, 16, 2048);
    add_lora(v, 16, 2048, 4096, 16);
    add_lora(v, 64, 2048, 4096, 0);
  } else if (set == "test") {
    // Same families, shrunk so the O(mnk) double-double reference stays cheap, plus one
    // long-k case (k=4096) where FP64 rounding error accumulation is largest.
    add_uniform(v, 256, 16);
    add_uniform(v, 256, 64);
    add_uniform(v, 32, 256);
    add_uniform_mnk(v, 16, 64, 64, 4096);
    add_uniform_mnk(v, 64, 7, 13, 33);     // odd sizes: k not a multiple of 32 (Kp padding)
    add_moe(v, 8, 2, 256, 256, 896, 1.2, "prefill");
    add_moe(v, 64, 6, 16, 512, 352, 1.2, "decode");   // many m_e = 0
    add_hetero(v, 32, 16, 512);
    add_lora(v, 8, 256, 1024, 0);
  } else {  // quick
    add_uniform(v, 1024, 32);
    add_moe(v, 64, 6, 1024, 2048, 1408, 1.2, "prefill");
    add_hetero(v, 256, 16, 2048);
    add_lora(v, 64, 2048, 4096, 0);
  }
  return v;
}

// ---------------------------------------------------------------- data generation
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__global__ void k_fill_ozaki(double* x, size_t n, double phi, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t b = seed + 3 * i * 0x9E3779B97F4A7C15ull;
    double u0 = (mix64(b + 0x9E3779B97F4A7C15ull) >> 11) * 0x1.0p-53;
    double v = u0 - 0.5;
    if (phi != 0.0) {
      double u1 = ((mix64(b + 2 * 0x9E3779B97F4A7C15ull) >> 11) + 1) * 0x1.0p-53;  // (0,1]
      double u2 = (mix64(b + 3 * 0x9E3779B97F4A7C15ull) >> 11) * 0x1.0p-53;
      double z = sqrt(-2.0 * log(u1)) * cospi(2.0 * u2);   // Box-Muller
      v *= exp(phi * z);
    }
    x[i] = v;
  }
}
__global__ void k_fill_const(double* x, size_t n, double v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = v;
}
void fill_ozaki_device(double* d_x, size_t count, double phi, uint64_t seed, cudaStream_t s) {
  if (!count) return;
  k_fill_ozaki<<<(unsigned)std::min<size_t>((count + 255) / 256, 65535), 256, 0, s>>>(d_x, count, phi, seed);
}
void fill_const_device(double* d_x, size_t count, double v, cudaStream_t s) {
  if (!count) return;
  k_fill_const<<<(unsigned)std::min<size_t>((count + 255) / 256, 65535), 256, 0, s>>>(d_x, count, v);
}

int alloc_workload(const Workload& w, double phi, double alpha, double beta, int pad,
                   WorkloadBuffers* b) {
  *b = WorkloadBuffers();
  size_t nA = 0, nB = 0, nC = 0;
  for (auto& s : w.shapes) {
    nA += (size_t)(s.m + pad) * s.k;
    nB += (size_t)(s.k + pad) * s.n;
    nC += (size_t)(s.m + pad) * s.n;
  }
  b->nA = nA; b->nB = nB; b->nC = nC;
  // +1 so zero-size workloads still get valid (non-null) pointers
  WL_CUDA(cudaMalloc(&b->dA, (nA + 1) * sizeof(double)));
  WL_CUDA(cudaMalloc(&b->dB, (nB + 1) * sizeof(double)));
  WL_CUDA(cudaMalloc(&b->dC, (nC + 1) * sizeof(double)));
  WL_CUDA(cudaMalloc(&b->dC0, (nC + 1) * sizeof(double)));
  uint64_t ds = w.seed * 0x2545F4914F6CDD1Dull + (uint64_t)(phi * 1000.0);
  fill_ozaki_device(b->dA, nA, phi, ds ^ 0xA);
  fill_ozaki_device(b->dB, nB, phi, ds ^ 0xB);
  fill_ozaki_device(b->dC0, nC, phi, ds ^ 0xC);
  WL_CUDA(cudaMemcpy(b->dC, b->dC0, nC * sizeof(double), cudaMemcpyDeviceToDevice));
  size_t oa = 0, ob = 0, oc = 0;
  for (auto& s : w.shapes) {
    ggemm_problem_t p;
    p.m = s.m; p.n = s.n; p.k = s.k;
    p.lda = std::max(1, s.m + pad); p.ldb = std::max(1, s.k + pad); p.ldc = std::max(1, s.m + pad);
    p.A = b->dA + oa; p.B = b->dB + ob; p.C = b->dC + oc;
    p.alpha = alpha; p.beta = beta;
    b->probs.push_back(p); b->offC.push_back(oc);
    oa += (size_t)(s.m + pad) * s.k; ob += (size_t)(s.k + pad) * s.n; oc += (size_t)(s.m + pad) * s.n;
  }
  WL_CUDA(cudaDeviceSynchronize());
  return 0;
}
void reset_C(WorkloadBuffers* b, cudaStream_t s) {
  cudaMemcpyAsync(b->dC, b->dC0, b->nC * sizeof(double), cudaMemcpyDeviceToDevice, s);
}
void free_workload(WorkloadBuffers* b) {
  cudaFree(b->dA); cudaFree(b->dB); cudaFree(b->dC); cudaFree(b->dC0);
  *b = WorkloadBuffers();
}

// ---------------------------------------------------------------- CLI helpers
static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out; std::string cur;
  for (char c : s) { if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) out.push_back(cur);
  return out;
}
std::vector<ggemm_path_t> parse_paths(const std::string& s) {
  std::vector<ggemm_path_t> all = {GGEMM_PATH_NATIVE_LOOP, GGEMM_PATH_NATIVE_GROUPED, GGEMM_PATH_EMU_LOOP,
                                   GGEMM_PATH_EMU_GROUPED, GGEMM_PATH_AUTO};
  if (s == "all") return all;
  std::vector<ggemm_path_t> out;
  for (auto& t : split_csv(s))
    for (auto p : all) if (t == ggemm_path_name(p)) out.push_back(p);
  return out;
}
std::vector<double> parse_doubles(const std::string& s) {
  std::vector<double> v; for (auto& t : split_csv(s)) v.push_back(atof(t.c_str())); return v;
}
std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v; for (auto& t : split_csv(s)) v.push_back(atoi(t.c_str())); return v;
}
