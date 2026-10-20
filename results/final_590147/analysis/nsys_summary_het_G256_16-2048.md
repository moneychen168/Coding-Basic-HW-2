# nsys summary: het_G256_16-2048
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 303 | 0 | 3027.7 | 2614.5 | 410.5 | 13.6 | 1078.0 | 0.0 | 2607.6 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1980.8 | 1967.4 | 13.5 | 0.7 | 13.8 | 0.0 | 29.8 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 4580 | 1 | 58,692 | 53,109 | 5574.9 | 9.5 | 40,826 | 1605.2 | 47,926 | 22.7 | 67.6 | 9.7 | 0.0 | 0.1 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 10,037 | 10,032 | 5.3 | 0.1 | 16.2 | 220.1 | 238.6 | 41.3 | 35.4 | 23.2 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1984.1 | 1971.6 | 12.8 | 0.6 | 12.9 | 40.8 | 61.5 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 11,216 | 11,206 | 10.6 | 0.1 | 15.9 | 228.0 | 246.4 | 47.1 | 32.0 | 20.8 | 0.0 | 0.1 | 0.0 |
| emu_loop_batched | 5 | 768 | 1 | 23,850 | 22,744 | 1104.4 | 4.6 | 2271.7 | 222.1 | 3340.0 | 52.8 | 24.6 | 22.5 | 0.0 | 0.1 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,345,594 | 449.88 | 57.6 |
| cudaLaunchKernel | 37,691 | 319,856 | 8.49 | 13.7 |
| cudaEventSynchronize | 77 | 293,811 | 3815.72 | 12.6 |
| cudaStreamCreateWithFlags | 1 | 170,524 | 170,524 | 7.3 |
| cudaStreamSynchronize | 21 | 98,900 | 4709.53 | 4.2 |
| cudaMalloc | 77 | 43,475 | 564.61 | 1.9 |
| cudaFree | 85 | 33,367 | 392.56 | 1.4 |
| cudaHostAlloc | 7 | 13,913 | 1987.52 | 0.6 |
| cudaLaunchKernelExC_v11060 | 1603 | 5851.1 | 3.65 | 0.3 |
| cudaFreeHost | 7 | 2879.8 | 411.40 | 0.1 |
| cuKernelGetName | 37,691 | 2540.6 | 0.07 | 0.1 |
| cudaDeviceSynchronize | 33 | 1810.3 | 54.86 | 0.1 |
| cuLaunchKernel | 266 | 829.4 | 3.12 | 0.0 |
| cudaMemcpyAsync | 56 | 338.4 | 6.04 | 0.0 |
| cudaEventRecord | 126 | 255.4 | 2.03 | 0.0 |
