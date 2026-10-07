// bench/app/app_common.h -- shared helpers for the FP64 application-level benchmarks (owner: agent-E).
// Only used by bench/app/*.cu. GEMMs are issued exclusively through include/ggemm.h.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "ggemm.h"
#include "../workloads.h"

#define APP_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static inline double app_median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static inline double app_mean(const std::vector<double>& v) {
  if (v.empty()) return NAN;
  double s = 0; for (double x : v) s += x; return s / v.size();
}
static inline double app_gamma(long long n) {   // n u / (1 - n u), u = 2^-53
  const double u = 0x1.0p-53; return (double)n * u / (1.0 - (double)n * u);
}
// Same per-path config convention as bench/bench_grouped.cu: adaptive moduli, FP64-level
// target, fallback only for the planner path.
static inline ggemm_config_t app_cfg(ggemm_path_t p, int num_moduli) {
  ggemm_config_t c; c.num_moduli = num_moduli; c.target_rel_err = 1e-15;
  c.allow_fallback = (p == GGEMM_PATH_AUTO) ? 1 : 0; return c;
}
static inline std::string app_gpu_name() {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    int cnt = -1; cudaError_t e2 = cudaGetDeviceCount(&cnt);
    const char* vis = getenv("CUDA_VISIBLE_DEVICES");
    fprintf(stderr, "cudaGetDeviceProperties failed: %s; cudaGetDeviceCount=%d (%s); CUDA_VISIBLE_DEVICES=%s\n",
            cudaGetErrorString(err), cnt, cudaGetErrorString(e2), vis ? vis : "(unset)");
    exit(1);
  }
  std::string g = prop.name; for (auto& c : g) if (c == ',') c = ' ';
  return g;
}

// ---------------------------------------------------------------- elementwise kernels
static __global__ void k_app_abs(const double* x, double* y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    y[i] = fabs(x[i]);
}
static __global__ void k_app_scale(double* x, size_t n, double s) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] *= s;
}
// Library-tuning environment recorded in every CSV row (';'-separated, no commas).
static inline std::string app_env_str() {
  const char* keys[] = {"GGEMM_REUSE", "GGEMM_REUSE_PROMOTE", "GGEMM_STATS_REPAIR", "GGEMM_CALIB_JSON"};
  std::string out;
  for (const char* k : keys) {
    const char* v = getenv(k);
    std::string sv = v ? v : "";
    for (auto& c : sv) if (c == ',') c = ';';
    if (!out.empty()) out += ';';
    out += std::string(k) + "=" + (v ? sv : "unset");
  }
  return out;
}
// Path names; emu_loop_batched (6) is named here because bench/workloads.h may not know it yet.
static inline const char* app_path_name(ggemm_path_t p) {
  if ((int)p == 6) return "emu_loop_batched";
  return ggemm_path_name(p);
}
// "all" = every path parse_paths knows (incl. emu_grouped_reuse = 5) plus GGEMM_PATH_EMU_LOOP_BATCHED
// (= 6, present in include/ggemm.h). Explicit lists may also name "emu_loop_batched".
static inline std::vector<ggemm_path_t> app_parse_paths(const std::string& s) {
  std::vector<ggemm_path_t> v = parse_paths(s);
  const ggemm_path_t p6 = GGEMM_PATH_EMU_LOOP_BATCHED;
  bool want6 = (s == "all" || s.find("emu_loop_batched") != std::string::npos);
  if (want6 && std::find(v.begin(), v.end(), p6) == v.end()) v.push_back(p6);
  return v;
}
static inline unsigned app_grid(size_t n) { return (unsigned)std::min<size_t>((n + 255) / 256, 65535); }
static inline void app_abs(const double* x, double* y, size_t n, cudaStream_t s) {
  if (n) k_app_abs<<<app_grid(n), 256, 0, s>>>(x, y, n);
}
static inline void app_scale(double* x, size_t n, double v, cudaStream_t s) {
  if (n) k_app_scale<<<app_grid(n), 256, 0, s>>>(x, n, v);
}

// ---------------------------------------------------------------- componentwise comparison
// out[0] = bits of max_i |x_i - r_i| / den_i  (non-negative doubles order like their bit patterns)
// out[1] = #entries where x_i or r_i is non-finite, or den_i == 0 with x_i != r_i
// sums[0] = sum (x-r)^2 over finite pairs, sums[1] = sum r^2 over every finite r_i (non-finite
//           x points stay in the denominator)   (Frobenius relative error, informational)
static __global__ void k_app_cmp(const double* x, const double* r, const double* den, size_t n,
                                 unsigned long long* out, double* sums) {
  double emax = 0, s0 = 0, s1 = 0; unsigned long long bad = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    if (!isfinite(r[i])) { ++bad; continue; }
    s1 += r[i] * r[i];
    if (!isfinite(x[i]) || !isfinite(den[i])) { ++bad; continue; }
    double d = fabs(x[i] - r[i]);
    if (den[i] > 0) emax = fmax(emax, d / den[i]);
    else if (d != 0) ++bad;
    s0 += d * d;
  }
  if (emax > 0) atomicMax(&out[0], (unsigned long long)__double_as_longlong(emax));
  if (bad) atomicAdd(&out[1], bad);
  if (s0 != 0) atomicAdd(&sums[0], s0);
  if (s1 != 0) atomicAdd(&sums[1], s1);
}
struct AppErr { double cw = NAN, rel_fro = NAN; unsigned long long bad = 0; };
// Untimed. scratch must hold 2 ull + 2 double (device).
static inline AppErr app_compare(const double* x, const double* r, const double* den, size_t n,
                                 unsigned long long* d_u, double* d_s, cudaStream_t s) {
  AppErr e;
  APP_CUDA(cudaMemsetAsync(d_u, 0, 2 * sizeof(unsigned long long), s));
  APP_CUDA(cudaMemsetAsync(d_s, 0, 2 * sizeof(double), s));
  if (n) k_app_cmp<<<std::min(app_grid(n), 1024u), 256, 0, s>>>(x, r, den, n, d_u, d_s);
  unsigned long long hu[2]; double hs[2];
  APP_CUDA(cudaMemcpyAsync(hu, d_u, sizeof hu, cudaMemcpyDeviceToHost, s));
  APP_CUDA(cudaMemcpyAsync(hs, d_s, sizeof hs, cudaMemcpyDeviceToHost, s));
  APP_CUDA(cudaStreamSynchronize(s));
  double cw; memcpy(&cw, &hu[0], sizeof cw);
  e.cw = cw; e.bad = hu[1]; e.rel_fro = hs[1] > 0 ? std::sqrt(hs[0] / hs[1]) : (hs[0] > 0 ? INFINITY : 0.0);
  return e;
}
