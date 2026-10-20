# nsys summary: nsys_new
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_grouped | 120 | 2 | 1 | 875.2 | 862.3 | 10.9 | 1.2 | 12.1 | 0.0 | 21.3 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| auto | 120 | 2 | 1 | 879.3 | 863.8 | 12.1 | 1.4 | 11.9 | 5.1 | 24.8 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,445,619 | 483.32 | 39.8 |
| cudaDeviceSynchronize | 268 | 493,153 | 1840.12 | 13.6 |
| cudaMemcpy | 96 | 488,522 | 5088.77 | 13.5 |
| cudaMalloc | 675 | 380,714 | 564.02 | 10.5 |
| cudaFree | 724 | 223,717 | 309.00 | 6.2 |
| cudaEventSynchronize | 624 | 216,595 | 347.11 | 6.0 |
| cudaStreamCreateWithFlags | 1 | 150,757 | 150,757 | 4.2 |
| cudaStreamSynchronize | 192 | 138,895 | 723.41 | 3.8 |
| cudaHostAlloc | 48 | 44,909 | 935.60 | 1.2 |
| cudaLaunchKernel | 7021 | 20,986 | 2.99 | 0.6 |
| cudaFreeHost | 48 | 12,556 | 261.59 | 0.3 |
| cudaLaunchKernelExC_v11060 | 1472 | 4981.0 | 3.38 | 0.1 |
| cudaMemcpyAsync | 792 | 3575.3 | 4.51 | 0.1 |
| cudaEventRecord | 1881 | 2058.6 | 1.09 | 0.1 |
| cuLaunchKernel | 369 | 1089.9 | 2.95 | 0.0 |
