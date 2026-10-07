// scripts/cuda_preflight.cu -- job-start GPU sanity check (owner: agent-B).
// Exit 0 only if CUDA initialises, >=1 device is visible, a context can be created and a
// trivial kernel runs. Some nodes have GPUs where NVML works but CUDA init fails.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k_ping(int* x) { *x = 42; }
int main() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n < 1) {
    fprintf(stderr, "[preflight] cudaGetDeviceCount: %s, count=%d\n", cudaGetErrorString(e), n); return 1;
  }
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p;
    if ((e = cudaSetDevice(d)) != cudaSuccess || (e = cudaGetDeviceProperties(&p, d)) != cudaSuccess) {
      fprintf(stderr, "[preflight] device %d: %s\n", d, cudaGetErrorString(e)); return 1;
    }
    int* x = nullptr; int h = 0;
    if ((e = cudaMalloc(&x, sizeof(int))) != cudaSuccess) { fprintf(stderr, "[preflight] dev %d malloc: %s\n", d, cudaGetErrorString(e)); return 1; }
    k_ping<<<1, 1>>>(x);
    if ((e = cudaMemcpy(&h, x, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess || h != 42) {
      fprintf(stderr, "[preflight] dev %d kernel: %s (h=%d)\n", d, cudaGetErrorString(e), h); return 1;
    }
    cudaFree(x);
    printf("[preflight] ok: device %d %s sm_%d%d\n", d, p.name, p.major, p.minor);
  }
  return 0;
}
