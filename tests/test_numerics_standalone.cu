// Standalone test of the Ozaki-II numerics (agent-N): prep -> naive INT8 GEMM -> reconstruct.
// Build (login node: compile only):
//   /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 \
//     -gencode arch=compute_120,code=sm_120 -Isrc -Iinclude \
//     tests/test_numerics_standalone.cu src/ozaki2_numerics.cu -o build/test_numerics_standalone
// Metrics (all against a CPU long double reference with Neumaier summation):
//   rel_cw  = max_ij |C - R| / |R|                     (componentwise relative, R != 0)
//   rel_abs = max_ij |C - R| / (|alpha| (|A||B|)_ij)   (relative to |A||B|)
//   rel_nrm = max_ij |C - R| / (|alpha| ||A(i,:)|| ||B(:,j)||) -- informational (oz2_error_bound)
//   gate    = max_ij |C - R| / (gamma_n (|alpha|(|A||B|)_ij + |beta||C0_ij|)),
//             n = k (+1 alpha != 1) (+1 beta != 0): the componentwise FP64 contract, must be <= 1
//   rep     = entries recomputed in FP64 by the certificate (oz2_read_counters)
// Exit code 0 iff every gate <= 1 and the residue checks pass.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "moduli.h"
#include "ozaki2_numerics.h"

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                         \
    }                                                                                  \
  } while (0)

// Reference INT8 GEMM: C32[l](i,j) = sum_k A8[l](i,k) * B8[l](j,k), deliberately simple.
__global__ void naive_i8_gemm(const oz2_problem* probs) {
  const oz2_problem& p = probs[blockIdx.z];
  const int l = blockIdx.y;
  if (l > p.num_moduli) return;  // planes 0..N (N = abs plane)
  const size_t total = (size_t)p.m * p.n;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    const int i = (int)(idx % p.m), j = (int)(idx / p.m);
    const int8_t* a = p.A8 + l * p.a8_mod_stride + (size_t)i * p.kp;
    const int8_t* b = p.B8 + l * p.b8_mod_stride + (size_t)j * p.kp;
    int32_t acc = 0;
    for (int k = 0; k < p.kp; ++k) acc += (int32_t)a[k] * (int32_t)b[k];
    p.C32[l * p.c32_mod_stride + idx] = acc;
  }
}

struct HostProb {
  int m, n, k, lda, ldb, ldc, N;
  double alpha, beta;
  std::vector<double> A, B, C0;   // column-major with leading dims
  std::vector<long double> R;     // reference alpha*AB + beta*C0, m x n col-major
  std::vector<double> absAB, nrm; // (|A||B|)_ij and ||a_i|| ||b_j||
};

static std::mt19937_64 rng(12345);

// GEMMul8-style generator: (U - 0.5) * exp(phi * Z)
static double gen(double phi) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> Z(0.0, 1.0);
  return (U(rng) - 0.5) * std::exp(phi * Z(rng));
}

static void make(HostProb& h, int m, int n, int k, int lda_pad, double phi, double alpha, double beta) {
  h.m = m; h.n = n; h.k = k; h.lda = m + lda_pad; h.ldb = k + 3; h.ldc = m + 1;
  h.alpha = alpha; h.beta = beta;
  h.A.assign((size_t)h.lda * k, 0.0);
  h.B.assign((size_t)h.ldb * n, 0.0);
  h.C0.assign((size_t)h.ldc * n, 0.0);
  for (int c = 0; c < k; ++c) for (int r = 0; r < m; ++r) h.A[r + (size_t)c * h.lda] = gen(phi);
  for (int c = 0; c < n; ++c) for (int r = 0; r < k; ++r) h.B[r + (size_t)c * h.ldb] = gen(phi);
  for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r) h.C0[r + (size_t)c * h.ldc] = gen(0.0);
}

static void reference(HostProb& h) {
  const int m = h.m, n = h.n, k = h.k;
  h.R.assign((size_t)m * n, 0.0L);
  h.absAB.assign((size_t)m * n, 0.0);
  h.nrm.assign((size_t)m * n, 0.0);
  std::vector<long double> ra(m, 0.0L), cb(n, 0.0L);
  for (int i = 0; i < m; ++i) for (int q = 0; q < k; ++q) { long double a = h.A[i + (size_t)q * h.lda]; ra[i] += a * a; }
  for (int j = 0; j < n; ++j) for (int q = 0; q < k; ++q) { long double b = h.B[q + (size_t)j * h.ldb]; cb[j] += b * b; }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      long double s = 0.0L, comp = 0.0L, sa = 0.0L;
      for (int q = 0; q < k; ++q) {
        long double t = (long double)h.A[i + (size_t)q * h.lda] * (long double)h.B[q + (size_t)j * h.ldb];
        long double u = s + t;  // Neumaier
        comp += (fabsl(s) >= fabsl(t)) ? (s - u) + t : (t - u) + s;
        s = u;
        sa += fabsl(t);
      }
      s += comp;
      h.R[i + (size_t)j * m] = (long double)h.alpha * s + (long double)h.beta * h.C0[i + (size_t)j * h.ldc];
      h.absAB[i + (size_t)j * m] = (double)sa;
      h.nrm[i + (size_t)j * m] = (double)sqrtl(ra[i] * cb[j]);
    }
}

struct Dev {
  std::vector<double*> dA, dB, dC;
  int8_t *A8 = nullptr, *B8 = nullptr;
  int32_t *C32 = nullptr, *rs = nullptr, *cs = nullptr;
  oz2_problem* dprobs = nullptr;
  std::vector<oz2_problem> hp;
  int max_m = 0, max_n = 0, max_N = 0;
};

static void setup(Dev& d, std::vector<HostProb>& hs) {
  size_t a8 = 0, b8 = 0, c32 = 0, rsz = 0, csz = 0;
  for (auto& h : hs) {
    int kp = (h.k + 31) / 32 * 32;
    const int NP = oz2_num_planes(h.N);
    a8 += (size_t)NP * h.m * kp; b8 += (size_t)NP * h.n * kp; c32 += (size_t)NP * h.m * h.n;
    rsz += OZ2_ROW_WS_PER_ROW * h.m; csz += OZ2_ROW_WS_PER_ROW * h.n;
  }
  CK(cudaMalloc(&d.A8, a8 + 16)); CK(cudaMalloc(&d.B8, b8 + 16)); CK(cudaMalloc(&d.C32, 4 * c32 + 16));
  CK(cudaMalloc(&d.rs, 4 * rsz + 16)); CK(cudaMalloc(&d.cs, 4 * csz + 16));
  CK(cudaMemset(d.A8, 0x5a, a8 + 16)); CK(cudaMemset(d.B8, 0x5a, b8 + 16));  // poison pads
  size_t oa = 0, ob = 0, oc = 0, orr = 0, occ = 0;
  for (auto& h : hs) {
    double *A, *B, *C;
    CK(cudaMalloc(&A, 8 * h.A.size())); CK(cudaMalloc(&B, 8 * h.B.size())); CK(cudaMalloc(&C, 8 * h.C0.size()));
    CK(cudaMemcpy(A, h.A.data(), 8 * h.A.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, h.B.data(), 8 * h.B.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(C, h.C0.data(), 8 * h.C0.size(), cudaMemcpyHostToDevice));
    d.dA.push_back(A); d.dB.push_back(B); d.dC.push_back(C);
    oz2_problem p{};
    p.m = h.m; p.n = h.n; p.k = h.k; p.kp = (h.k + 31) / 32 * 32; p.num_moduli = h.N;
    p.lda = h.lda; p.ldb = h.ldb; p.ldc = h.ldc; p.A = A; p.B = B; p.C = C;
    p.alpha = h.alpha; p.beta = h.beta;
    p.A8 = d.A8 + oa; p.a8_mod_stride = (size_t)h.m * p.kp;
    p.B8 = d.B8 + ob; p.b8_mod_stride = (size_t)h.n * p.kp;
    p.C32 = d.C32 + oc; p.c32_mod_stride = (size_t)h.m * h.n;
    p.row_shift = d.rs + orr; p.col_shift = d.cs + occ;
    const int NP = oz2_num_planes(h.N);
    oa += (size_t)NP * p.a8_mod_stride; ob += (size_t)NP * p.b8_mod_stride;
    oc += (size_t)NP * p.c32_mod_stride; orr += OZ2_ROW_WS_PER_ROW * h.m; occ += OZ2_ROW_WS_PER_ROW * h.n;
    d.hp.push_back(p);
    d.max_m = std::max(d.max_m, h.m); d.max_n = std::max(d.max_n, h.n); d.max_N = std::max(d.max_N, h.N);
  }
  CK(cudaMalloc(&d.dprobs, sizeof(oz2_problem) * d.hp.size()));
  CK(cudaMemcpy(d.dprobs, d.hp.data(), sizeof(oz2_problem) * d.hp.size(), cudaMemcpyHostToDevice));
}

static void teardown(Dev& d) {
  for (auto p : d.dA) cudaFree(p);
  for (auto p : d.dB) cudaFree(p);
  for (auto p : d.dC) cudaFree(p);
  cudaFree(d.A8); cudaFree(d.B8); cudaFree(d.C32); cudaFree(d.rs); cudaFree(d.cs); cudaFree(d.dprobs);
}

static void run(Dev& d) {
  const int num = (int)d.hp.size();
  if (oz2_prep_grouped(d.dprobs, num, d.max_m, d.max_n, 0)) { fprintf(stderr, "prep failed\n"); exit(2); }
  int gx = (int)std::min<size_t>(((size_t)d.max_m * d.max_n + 255) / 256, 4096);
  naive_i8_gemm<<<dim3(gx, oz2_num_planes(d.max_N), num), 256>>>(d.dprobs);
  CK(cudaGetLastError());
  if (oz2_reconstruct_grouped(d.dprobs, num, d.max_m, d.max_n, 0)) { fprintf(stderr, "recon failed\n"); exit(2); }
  CK(cudaDeviceSynchronize());
}

struct Err { double cw = 0, ab = 0, nr = 0, gate = 0; bool beta_ok = true; };

static double gamma_n(const HostProb& h) {
  double n = h.k + (h.alpha != 1.0) + (h.beta != 0.0);
  if (n < 1) n = 1;
  const double u = std::ldexp(1.0, -53);
  return n * u / (1.0 - n * u);
}
static unsigned long long take_repairs() {
  unsigned long long r = 0;
  oz2_read_counters(nullptr, &r, 1, 0);
  return r;
}

static Err check(const Dev& d, int idx, const HostProb& h) {
  std::vector<double> C(h.C0.size());
  CK(cudaMemcpy(C.data(), d.dC[idx], 8 * C.size(), cudaMemcpyDeviceToHost));
  Err e;
  for (int j = 0; j < h.n; ++j) {
    for (int i = 0; i < h.m; ++i) {
      long double r = h.R[i + (size_t)j * h.m];
      double c = C[i + (size_t)j * h.ldc];
      // remove the beta*C0 part's own rounding (<= 2u|beta C0|) from the error budget
      long double err = fabsl((long double)c - r);
      long double slack = 4.0L * ldexpl(1.0L, -53) * (fabsl((long double)h.beta * h.C0[i + (size_t)j * h.ldc]) + fabsl(r));
      long double err_ab = err > slack ? err - slack : 0.0L;
      if (r != 0) e.cw = std::max(e.cw, (double)(err / fabsl(r)));
      double sa = fabs(h.alpha) * h.absAB[i + (size_t)j * h.m];
      double sn = fabs(h.alpha) * h.nrm[i + (size_t)j * h.m];
      if (sa > 0) e.ab = std::max(e.ab, (double)(err_ab / sa));
      if (sn > 0) e.nr = std::max(e.nr, (double)(err_ab / sn));
      else if (err_ab > 0) e.nr = INFINITY;
      long double den = gamma_n(h) * ((long double)sa + fabsl((long double)h.beta * h.C0[i + (size_t)j * h.ldc]));
      if (den > 0) e.gate = std::max(e.gate, (double)(err / den));
      else if (err > 0) e.gate = INFINITY;
    }
    for (int i = h.m; i < h.ldc; ++i)  // padding rows of C must be untouched
      if (C[i + (size_t)j * h.ldc] != h.C0[i + (size_t)j * h.ldc]) e.beta_ok = false;
  }
  return e;
}

// Exact residue check on the host: trunc(a*2^s) has <= 53 significant bits -> exact in
// long double; fmodl is exact.
static bool check_residues(const Dev& d, int idx, const HostProb& h) {
  const oz2_problem& p = d.hp[idx];
  const int NP = oz2_num_planes(h.N);
  std::vector<int8_t> A8((size_t)NP * p.a8_mod_stride), B8((size_t)NP * p.b8_mod_stride);
  std::vector<int32_t> rs(OZ2_ROW_WS_PER_ROW * h.m), cs(OZ2_ROW_WS_PER_ROW * h.n);
  CK(cudaMemcpy(A8.data(), p.A8, A8.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(B8.data(), p.B8, B8.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rs.data(), p.row_shift, 4 * rs.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cs.data(), p.col_shift, 4 * cs.size(), cudaMemcpyDeviceToHost));
  auto absval = [](double a, int sg) {
    return sg == OZ2_SHIFT_ZERO ? 0 : (int)truncl(ldexpl(fabsl((long double)a), sg));
  };
  oz2_crt_tables t; oz2_build_crt_tables(&t);
  const long double T = t.T[h.N];
  long bad = 0;
  auto sym = [](long double x, int pm) {
    long double r = fmodl(x, (long double)pm);
    if (r < 0) r += pm;
    int ri = (int)r;
    if (ri > (pm - 1) / 2) ri -= pm;
    return ri;
  };
  for (int i = 0; i < h.m; ++i) {
    long double n2 = 0;
    for (int q = 0; q < p.kp; ++q) {
      long double x = q < h.k ? truncl(ldexpl((long double)h.A[i + (size_t)q * h.lda], rs[i])) : 0.0L;
      n2 += x * x;
      for (int l = 0; l < h.N; ++l)
        if (A8[l * p.a8_mod_stride + (size_t)i * p.kp + q] != sym(x, t.p[l])) ++bad;
      int av = q < h.k ? absval(h.A[i + (size_t)q * h.lda], rs[h.m + i]) : 0;
      if (av < 0 || av > 127 || A8[h.N * p.a8_mod_stride + (size_t)i * p.kp + q] != av) ++bad;
    }
    if (sqrtl(n2) > T) { ++bad; fprintf(stderr, "row %d norm %Lg > T %Lg\n", i, sqrtl(n2), T); }
  }
  for (int j = 0; j < h.n; ++j) {
    long double n2 = 0;
    for (int q = 0; q < p.kp; ++q) {
      long double x = q < h.k ? truncl(ldexpl((long double)h.B[q + (size_t)j * h.ldb], cs[j])) : 0.0L;
      n2 += x * x;
      for (int l = 0; l < h.N; ++l)
        if (B8[l * p.b8_mod_stride + (size_t)j * p.kp + q] != sym(x, t.p[l])) ++bad;
      int bv = q < h.k ? absval(h.B[q + (size_t)j * h.ldb], cs[h.n + j]) : 0;
      if (bv < 0 || bv > 127 || B8[h.N * p.b8_mod_stride + (size_t)j * p.kp + q] != bv) ++bad;
    }
    if (sqrtl(n2) > T) { ++bad; fprintf(stderr, "col %d norm %Lg > T %Lg\n", j, sqrtl(n2), T); }
  }
  printf("residue check (m=%d n=%d k=%d N=%d): %ld mismatches\n", h.m, h.n, h.k, h.N, bad);
  return bad == 0;
}

int main() {
  int fails = 0;
  if (int e = oz2_init_constants()) { fprintf(stderr, "oz2_init_constants = %d\n", e); return 2; }
  {
    oz2_crt_tables t; oz2_build_crt_tables(&t);
    printf("N  log2P    bound(k=1024)  choose(k,1e-15)  [k=256 k=1024 k=4096 k=16384 k=65536]\n");
    for (int N = 8; N <= GGEMM_MAX_MODULI; ++N) printf("%2d %7.2f  %.3e\n", N, t.log2P[N], oz2_error_bound(1024, N));
    printf("choose(1e-15): k=256:%d 1024:%d 4096:%d 16384:%d 65536:%d 131072:%d\n",
           oz2_choose_num_moduli(256, 1e-15), oz2_choose_num_moduli(1024, 1e-15), oz2_choose_num_moduli(4096, 1e-15),
           oz2_choose_num_moduli(16384, 1e-15), oz2_choose_num_moduli(65536, 1e-15), oz2_choose_num_moduli(131072, 1e-15));
  }

  // 1) exact residue check, small odd shapes, a few N
  for (int N : {1, 8, 13, 20}) {
    std::vector<HostProb> hs(1);
    make(hs[0], 37, 29, 100, 5, 2.0, 1.0, 0.0);
    hs[0].N = N;
    Dev d; setup(d, hs);
    if (oz2_prep_grouped(d.dprobs, 1, d.max_m, d.max_n, 0)) return 2;
    CK(cudaDeviceSynchronize());
    if (!check_residues(d, 0, hs[0])) ++fails;
    teardown(d);
  }

  // 2) sweep N = 8..20 x phi, single problem per launch (same data reused across N)
  const int shapes[][3] = {{16, 16, 16}, {64, 64, 64}, {128, 96, 1000}, {64, 64, 4096}};
  for (auto& sh : shapes)
    for (double phi : {0.0, 0.5, 1.0, 2.0, 4.0}) {
      HostProb base;
      make(base, sh[0], sh[1], sh[2], 3, phi, 1.0, 0.0);
      reference(base);
      // FP64 naive dot for context
      double fp64_cw = 0;
      for (int j = 0; j < base.n; ++j)
        for (int i = 0; i < base.m; ++i) {
          double s = 0;
          for (int q = 0; q < base.k; ++q) s += base.A[i + (size_t)q * base.lda] * base.B[q + (size_t)j * base.ldb];
          long double r = base.R[i + (size_t)j * base.m];
          if (r != 0) fp64_cw = std::max(fp64_cw, (double)(fabsl(s - r) / fabsl(r)));
        }
      printf("\nshape m=%d n=%d k=%d phi=%.1f   (naive FP64 dot rel_cw=%.3e)\n", sh[0], sh[1], sh[2], phi, fp64_cw);
      printf(" N   rel_cw      rel_abs     rel_nrm     bound       gate        rep     ok\n");
      for (int N = 8; N <= GGEMM_MAX_MODULI; ++N) {
        std::vector<HostProb> hs(1, base);
        hs[0].N = N;
        take_repairs();
        Dev d; setup(d, hs); run(d);
        unsigned long long rp = take_repairs();
        Err e = check(d, 0, hs[0]);
        double bnd = oz2_error_bound(base.k, N);
        bool ok = e.gate <= 1.0 && e.beta_ok;
        if (!ok) ++fails;
        printf("%2d  %.3e  %.3e  %.3e  %.3e  %.3e  %-6llu  %s\n", N, e.cw, e.ab, e.nr, bnd, e.gate, rp, ok ? "ok" : "FAIL");
        teardown(d);
      }
    }

  // 3) one grouped launch with heterogeneous problems
  {
    struct S { int m, n, k, N; double phi, alpha, beta; int special; };
    const S spec[] = {
        {200, 150, 777, 20, 1.0, 1.0, 0.0, 0},
        {1, 300, 64, 16, 0.5, -2.5, 1.0, 0},
        {300, 1, 129, 12, 2.0, 0.75, -0.5, 0},
        {77, 55, 1, 8, 1.0, 1.0, 2.0, 0},
        {64, 48, 512, 18, 1.0, 1.0, 0.0, 1},   // extreme exponents: A*2^-700, B*2^650 (rescan path)
        {40, 33, 300, 14, 1.0, 1.0, 0.0, 2},   // zero row and zero column
        {33, 65, 2048, 10, 4.0, 1.0, 0.25, 0},
        {48, 48, 256, 15, 1.0, 1.0, 0.0, 3},   // adversarial cancellation: B(:,j) orthogonal-ish to A rows
        {16, 16, 16, 15, 2.0, 1.0, 0.0, 0},    // the 0926 failing class (k=16, phi=2, N=15)
    };
    std::vector<HostProb> hs;
    for (auto& s : spec) {
      HostProb h;
      make(h, s.m, s.n, s.k, 7, s.phi, s.alpha, s.beta);
      h.N = s.N;
      if (s.special == 1) {
        for (auto& a : h.A) a = std::ldexp(a, -700);
        for (auto& b : h.B) b = std::ldexp(b, 650);
      } else if (s.special == 2) {
        for (int q = 0; q < h.k; ++q) h.A[3 + (size_t)q * h.lda] = 0.0;
        for (int q = 0; q < h.k; ++q) h.B[q + (size_t)5 * h.ldb] = 0.0;
      } else if (s.special == 3) {  // B(:,j) = x_j * (A(0,:) shifted) - pairs that cancel in row 0
        for (int j = 0; j < h.n; ++j)
          for (int q = 0; q + 1 < h.k; q += 2) {
            h.B[q + (size_t)j * h.ldb] = h.A[0 + (size_t)(q + 1) * h.lda];
            h.B[q + 1 + (size_t)j * h.ldb] = -h.A[0 + (size_t)q * h.lda] * (1.0 + std::ldexp((double)j, -40));
          }
      }
      reference(h);
      hs.push_back(std::move(h));
    }
    take_repairs();
    Dev d; setup(d, hs); run(d);
    printf("\ngrouped launch (%zu problems, 1 prep + 1 recon launch), total repaired entries=%llu\n", hs.size(),
           take_repairs());
    printf(" m    n    k     N   rel_cw      rel_abs     rel_nrm     bound       gate        ok\n");
    for (size_t g = 0; g < hs.size(); ++g) {
      Err e = check(d, (int)g, hs[g]);
      double bnd = oz2_error_bound(hs[g].k, hs[g].N);
      bool ok = e.gate <= 1.0 && e.beta_ok;
      if (!ok) ++fails;
      printf("%4d %4d %5d %3d  %.3e  %.3e  %.3e  %.3e  %.3e  %s\n", hs[g].m, hs[g].n, hs[g].k, hs[g].N, e.cw, e.ab, e.nr,
             bnd, e.gate, ok ? "ok" : "FAIL");
    }
    teardown(d);
  }

  // 4) non-finite handling: NaN in A row 2, +Inf in B column 4 -> those entries non-finite
  //    (computed by the FP64 repair path, like a native dot), rest meets the gate
  {
    std::vector<HostProb> hs(1);
    make(hs[0], 50, 40, 200, 2, 1.0, 1.0, 0.0);
    hs[0].N = 16;
    reference(hs[0]);
    hs[0].A[2 + (size_t)17 * hs[0].lda] = NAN;
    hs[0].B[33 + (size_t)4 * hs[0].ldb] = INFINITY;
    int cnt0 = 0;
    oz2_nonfinite_count(&cnt0, 1, 0);
    Dev d; setup(d, hs); run(d);
    int cnt = -1;
    oz2_nonfinite_count(&cnt, 1, 0);
    std::vector<double> C(hs[0].C0.size());
    CK(cudaMemcpy(C.data(), d.dC[0], 8 * C.size(), cudaMemcpyDeviceToHost));
    long badnan = 0; double gate = 0;
    const HostProb& h = hs[0];
    for (int j = 0; j < h.n; ++j)
      for (int i = 0; i < h.m; ++i) {
        double c = C[i + (size_t)j * h.ldc];
        if (i == 2 || j == 4) { if (std::isfinite(c)) ++badnan; continue; }
        long double err = fabsl((long double)c - h.R[i + (size_t)j * h.m]);
        gate = std::max(gate, (double)(err / (gamma_n(h) * h.absAB[i + (size_t)j * h.m])));
      }
    bool ok = cnt == 2 && badnan == 0 && gate <= 1.0;
    if (!ok) ++fails;
    printf("\nnon-finite: counter=%d (expect 2), finite entries in poisoned row/col=%ld, gate(rest)=%.3e %s\n",
           cnt, badnan, gate, ok ? "ok" : "FAIL");
    teardown(d);
  }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
