// bench/calib/calib.cu — AUTO planner cost-model calibration microbenchmarks (agent-C).
// Output: CSV on stdout. Lines starting with '#' are metadata.
//   # gpu,<name>
//   # sm_count,<n>
//   # cc,<major>.<minor>
//   kind,param,m,n,k,bytes,iters,time_us,value,unit
// kinds:
//   launch_stream   : empty kernel, back-to-back in one stream; time_us = per-launch amortized
//   launch_sync     : empty kernel + cudaStreamSynchronize; time_us = per round trip
//   h2d_pinned      : cudaMemcpyAsync H2D from pinned host (+ stream sync), bytes = size; time_us per copy
//   h2d_pinned_stream : same, back-to-back without sync between copies; time_us amortized
//   dgemm           : cublasDgemm m=n=k; value = GFLOP/s
//   i8gemm          : cublasLt INT8xINT8->INT32 (TN, K contiguous); value = GOP/s (2mnk)
//   bw_copy         : y[i]=x[i] double; bytes = 2*N*8 (read+write); value = GB/s
//   bw_read         : reduction read of x; bytes = N*8; value = GB/s
// Build: nvcc -O3 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 calib.cu -lcublas -lcublasLt -o calib
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS %s @%s:%d: status %d\n", #x, __FILE__, __LINE__, (int)s_); exit(1);} } while (0)

__global__ void k_empty() {}

__global__ void k_copy(const double* __restrict__ x, double* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) y[i] = x[i];
}
__global__ void k_read(const double* __restrict__ x, double* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  double s = 0.0;
  for (; i < n; i += st) s += x[i];
  if (s == 12345.678) out[0] = s;   // practically never; keeps the load alive
}
__global__ void k_fill_i8(int8_t* p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) p[i] = (int8_t)((i * 2654435761u >> 13) % 255 - 127);
}
__global__ void k_fill_f64(double* p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) p[i] = (double)((i * 2654435761u >> 13) % 1000) * 1e-3 - 0.5;
}

static void row(const char* kind, const char* param, long m, long n, long k, double bytes, int iters,
                double t_us, double value, const char* unit) {
  printf("%s,%s,%ld,%ld,%ld,%.0f,%d,%.4f,%.4f,%s\n", kind, param, m, n, k, bytes, iters, t_us, value, unit);
}

// Time `fn` (which enqueues `iters` ops on s) with events; repeat `reps` times, return median us per op.
template <class F>
static double time_med(cudaStream_t s, int iters, int reps, F fn) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  fn(); CK(cudaStreamSynchronize(s));  // warm-up
  std::vector<double> v;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a, s)); fn(); CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms * 1e3 / iters);
  }
  std::sort(v.begin(), v.end());
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  int maxdim = argc > 1 ? atoi(argv[1]) : 4096;
  int dev = 0; CK(cudaSetDevice(dev));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
  printf("# gpu,%s\n# sm_count,%d\n# cc,%d.%d\n", pr.name, pr.multiProcessorCount, pr.major, pr.minor);
  printf("kind,param,m,n,k,bytes,iters,time_us,value,unit\n");
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // ---------- (a) launch overhead ----------
  {
    const int IT = 2000;
    double t = time_med(s, IT, 7, [&] { for (int i = 0; i < IT; ++i) k_empty<<<1, 32, 0, s>>>(); });
    row("launch_stream", "1x32", 0, 0, 0, 0, IT, t, t, "us");
    t = time_med(s, IT, 7, [&] { for (int i = 0; i < IT; ++i) k_empty<<<1024, 256, 0, s>>>(); });
    row("launch_stream", "1024x256", 0, 0, 0, 0, IT, t, t, "us");
    // launch + sync round trip (host wall clock via events is not valid across sync; use events around each)
    const int RT = 500;
    std::vector<double> v;
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < RT; ++i) {
      CK(cudaEventRecord(a, s)); k_empty<<<1, 32, 0, s>>>(); CK(cudaEventRecord(b, s));
      CK(cudaStreamSynchronize(s));
      float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms * 1e3);
    }
    std::sort(v.begin(), v.end());
    row("launch_sync", "1x32_gpu_span", 0, 0, 0, 0, RT, v[RT / 2], v[RT / 2], "us");
    // pinned H2D small copies
    const size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    void* h; CK(cudaMallocHost(&h, 1 << 20)); void* d; CK(cudaMalloc(&d, 1 << 20));
    memset(h, 1, 1 << 20);
    for (size_t sz : sizes) {
      const int N = 500;
      double ts = time_med(s, N, 7, [&] { for (int i = 0; i < N; ++i) CK(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, s)); });
      row("h2d_pinned_stream", "", 0, 0, 0, (double)sz, N, ts, sz / ts * 1e-3, "GB/s");
      // isolated: copy + sync each time, measured as GPU span between events
      std::vector<double> w;
      for (int i = 0; i < 200; ++i) {
        CK(cudaEventRecord(a, s)); CK(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, s)); CK(cudaEventRecord(b, s));
        CK(cudaStreamSynchronize(s));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); w.push_back(ms * 1e3);
      }
      std::sort(w.begin(), w.end());
      row("h2d_pinned", "", 0, 0, 0, (double)sz, 200, w[100], sz / w[100] * 1e-3, "GB/s");
    }
    CK(cudaFreeHost(h)); CK(cudaFree(d));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  }

  std::vector<int> dims;
  for (int d = 32; d <= maxdim; d *= 2) { dims.push_back(d); if (d * 3 / 2 <= maxdim && d >= 64) dims.push_back(d * 3 / 2); }
  std::sort(dims.begin(), dims.end());
  size_t M = (size_t)maxdim;

  // ---------- (b) cublasDgemm ----------
  {
    cublasHandle_t hb; CB(cublasCreate(&hb)); CB(cublasSetStream(hb, s));
    double *A, *B, *C; CK(cudaMalloc(&A, M * M * 8)); CK(cudaMalloc(&B, M * M * 8)); CK(cudaMalloc(&C, M * M * 8));
    k_fill_f64<<<1024, 256, 0, s>>>(A, M * M); k_fill_f64<<<1024, 256, 0, s>>>(B, M * M);
    CK(cudaMemsetAsync(C, 0, M * M * 8, s));
    const double al = 1.0, be = 0.0;
    for (int d : dims) {
      double fl = 2.0 * d * d * d;
      int it = std::max(2, std::min(200, (int)(2e10 / fl)));
      double t = time_med(s, it, 5, [&] { for (int i = 0; i < it; ++i)
        CB(cublasDgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d, &al, A, d, B, d, &be, C, d)); });
      row("dgemm", "NN", d, d, d, 0, it, t, fl / t * 1e-3, "GFLOP/s");
    }
    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CB(cublasDestroy(hb));
  }

  // ---------- (c) cublasLt INT8 -> INT32 ----------
  {
    cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
    int8_t *A, *B; int32_t* C;
    CK(cudaMalloc(&A, M * M)); CK(cudaMalloc(&B, M * M)); CK(cudaMalloc(&C, M * M * 4));
    k_fill_i8<<<1024, 256, 0, s>>>(A, M * M); k_fill_i8<<<1024, 256, 0, s>>>(B, M * M);
    size_t wsb = 64ull << 20; void* ws; CK(cudaMalloc(&ws, wsb));
    const int32_t al = 1, be = 0;
    for (int d : dims) {
      // Layout matches DESIGN.md: A m x Kp row-major == col-major Kp x m (opA = T); B col-major Kp x n (opB = N);
      // C col-major m x n.
      cublasLtMatmulDesc_t op; CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
      cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
      CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
      CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
      cublasLtMatrixLayout_t la, lb, lc;
      CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, d, d, d));
      CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, d, d, d));
      CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, d, d, d));
      cublasLtMatmulPreference_t pf; CB(cublasLtMatmulPreferenceCreate(&pf));
      CB(cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
      cublasLtMatmulHeuristicResult_t hr; int nr = 0;
      cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pf, 1, &hr, &nr);
      double fl = 2.0 * d * d * d;
      if (st != CUBLAS_STATUS_SUCCESS || nr == 0) {
        row("i8gemm", "TN_noalgo", d, d, d, 0, 0, -1, -1, "GOP/s");
      } else {
        int it = std::max(2, std::min(500, (int)(1e11 / fl)));
        double t = time_med(s, it, 5, [&] { for (int i = 0; i < it; ++i)
          CB(cublasLtMatmul(lt, op, &al, A, la, B, lb, &be, C, lc, C, lc, &hr.algo, ws, wsb, s)); });
        row("i8gemm", "TN", d, d, d, 0, it, t, fl / t * 1e-3, "GOP/s");
      }
      cublasLtMatmulPreferenceDestroy(pf);
      cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
    }
    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(ws)); CB(cublasLtDestroy(lt));
  }

  // ---------- (d) DRAM bandwidth ----------
  {
    size_t maxn = (size_t)64 << 20;   // 64M doubles = 512 MB per buffer
    double *x, *y; CK(cudaMalloc(&x, maxn * 8)); CK(cudaMalloc(&y, maxn * 8));
    k_fill_f64<<<1024, 256, 0, s>>>(x, maxn);
    int blocks = pr.multiProcessorCount * 8;
    for (size_t n = (size_t)1 << 16; n <= maxn; n <<= 2) {
      int it = (int)std::max<size_t>(5, std::min<size_t>(1000, ((size_t)1 << 30) / (n * 8)));
      double t = time_med(s, it, 5, [&] { for (int i = 0; i < it; ++i) k_copy<<<blocks, 256, 0, s>>>(x, y, n); });
      double by = 2.0 * n * 8;
      row("bw_copy", "", 0, 0, 0, by, it, t, by / t * 1e-3, "GB/s");
      t = time_med(s, it, 5, [&] { for (int i = 0; i < it; ++i) k_read<<<blocks, 256, 0, s>>>(x, y, n); });
      by = 1.0 * n * 8;
      row("bw_read", "", 0, 0, 0, by, it, t, by / t * 1e-3, "GB/s");
    }
    CK(cudaFree(x)); CK(cudaFree(y));
  }
  CK(cudaStreamDestroy(s));
  fflush(stdout);
  return 0;
}
