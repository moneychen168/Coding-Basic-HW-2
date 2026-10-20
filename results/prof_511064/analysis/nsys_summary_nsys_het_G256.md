# nsys summary: nsys_het_G256
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 303 | 0 | 3002.2 | 2611.9 | 389.9 | 13.0 | 1091.2 | 0.0 | 2613.3 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1975.2 | 1962.2 | 13.5 | 0.7 | 13.9 | 0.0 | 29.6 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 4580 | 1 | 71,621 | 66,487 | 5134.4 | 7.2 | 51,222 | 1563.4 | 58,232 | 37.2 | 54.5 | 8.2 | 0.0 | 0.1 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 9640.2 | 9634.0 | 5.5 | 0.1 | 15.9 | 213.5 | 232.8 | 39.1 | 36.9 | 23.9 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1973.6 | 1961.4 | 12.7 | 0.6 | 12.9 | 40.2 | 61.0 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 10,487 | 10,479 | 10.4 | 0.1 | 15.8 | 221.9 | 240.6 | 44.0 | 33.9 | 22.0 | 0.0 | 0.2 | 0.0 |
| emu_loop_batched | 5 | 768 | 1 | 36,945 | 35,901 | 1040.2 | 2.8 | 2277.4 | 220.1 | 3368.9 | 68.9 | 15.9 | 15.2 | 0.0 | 0.0 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,528,034 | 510.88 | 56.6 |
| cudaLaunchKernel | 37,691 | 404,682 | 10.74 | 15.0 |
| cudaEventSynchronize | 77 | 367,752 | 4776.00 | 13.6 |
| cudaStreamCreateWithFlags | 1 | 177,407 | 177,407 | 6.6 |
| cudaStreamSynchronize | 21 | 117,654 | 5602.57 | 4.4 |
| cudaMalloc | 77 | 41,671 | 541.18 | 1.5 |
| cudaFree | 85 | 31,970 | 376.12 | 1.2 |
| cudaHostAlloc | 7 | 10,901 | 1557.35 | 0.4 |
| cudaLaunchKernelExC_v11060 | 1603 | 6886.0 | 4.30 | 0.3 |
| cudaFreeHost | 7 | 5315.8 | 759.41 | 0.2 |
| cuKernelGetName | 37,691 | 2534.2 | 0.07 | 0.1 |
| cudaDeviceSynchronize | 33 | 1794.3 | 54.37 | 0.1 |
| cuLaunchKernel | 266 | 857.9 | 3.23 | 0.0 |
| cudaMemcpyAsync | 56 | 321.0 | 5.73 | 0.0 |
| cudaEventRecord | 126 | 256.3 | 2.03 | 0.0 |
