# nsys summary: het_G256_16-2048
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 303 | 0 | 2986.5 | 2606.8 | 379.4 | 12.7 | 1016.6 | 0.0 | 2544.1 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1971.0 | 1958.1 | 13.0 | 0.7 | 13.6 | 0.0 | 27.9 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 4580 | 1 | 58,097 | 53,086 | 5018.8 | 8.6 | 40,217 | 1664.7 | 47,612 | 22.6 | 67.6 | 9.7 | 0.0 | 0.1 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 10,036 | 10,031 | 5.0 | 0.0 | 15.6 | 203.2 | 221.4 | 41.3 | 35.4 | 23.2 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1984.4 | 1969.4 | 13.0 | 0.7 | 13.8 | 44.7 | 66.1 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 11,186 | 11,175 | 5.4 | 0.0 | 21.5 | 241.8 | 265.6 | 47.2 | 31.8 | 20.8 | 0.0 | 0.1 | 0.0 |
| emu_loop_batched | 5 | 768 | 1 | 23,752 | 22,751 | 997.9 | 4.2 | 2144.3 | 227.2 | 3213.5 | 52.7 | 24.7 | 22.5 | 0.0 | 0.1 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,399,540 | 467.92 | 59.0 |
| cudaLaunchKernel | 37,691 | 315,005 | 8.36 | 13.3 |
| cudaEventSynchronize | 77 | 292,994 | 3805.11 | 12.4 |
| cudaStreamCreateWithFlags | 1 | 158,796 | 158,796 | 6.7 |
| cudaStreamSynchronize | 21 | 98,598 | 4695.12 | 4.2 |
| cudaMalloc | 77 | 43,111 | 559.89 | 1.8 |
| cudaFree | 85 | 34,631 | 407.43 | 1.5 |
| cudaHostAlloc | 7 | 14,101 | 2014.38 | 0.6 |
| cudaLaunchKernelExC_v11060 | 1603 | 5504.1 | 3.43 | 0.2 |
| cudaFreeHost | 7 | 3022.5 | 431.79 | 0.1 |
| cuKernelGetName | 37,691 | 2501.6 | 0.07 | 0.1 |
| cudaDeviceSynchronize | 33 | 1836.4 | 55.65 | 0.1 |
| cuLaunchKernel | 266 | 789.7 | 2.97 | 0.0 |
| cudaMemcpyAsync | 56 | 409.7 | 7.32 | 0.0 |
| cudaEventRecord | 126 | 285.9 | 2.27 | 0.0 |
