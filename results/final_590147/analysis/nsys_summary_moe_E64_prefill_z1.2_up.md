# nsys summary: moe_E64_prefill_z1.2_up
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 117 | 0 | 1671.3 | 1488.4 | 182.9 | 10.9 | 407.5 | 0.0 | 880.5 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1588.6 | 1573.0 | 12.3 | 0.8 | 13.3 | 0.0 | 22.9 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 1152 | 1 | 43,255 | 41,526 | 1723.1 | 4.0 | 3833.0 | 439.6 | 5675.3 | 42.6 | 54.1 | 3.3 | 0.0 | 0.0 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 6767.3 | 6761.7 | 5.6 | 0.1 | 16.6 | 44.1 | 65.0 | 63.7 | 29.1 | 7.0 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1584.5 | 1572.8 | 12.4 | 0.8 | 12.3 | 11.6 | 30.9 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 6679.9 | 6674.7 | 5.2 | 0.1 | 16.4 | 49.1 | 68.9 | 63.3 | 29.5 | 7.1 | 0.0 | 0.1 | 0.0 |
| emu_loop_batched | 5 | 192 | 1 | 22,414 | 22,100 | 311.3 | 1.4 | 591.8 | 62.5 | 877.5 | 80.5 | 13.5 | 6.0 | 0.0 | 0.0 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,577,797 | 527.51 | 64.2 |
| cudaEventSynchronize | 77 | 386,713 | 5022.24 | 15.7 |
| cudaStreamCreateWithFlags | 1 | 155,758 | 155,758 | 6.3 |
| cudaStreamSynchronize | 21 | 120,420 | 5734.31 | 4.9 |
| cudaMalloc | 77 | 87,527 | 1136.71 | 3.6 |
| cudaLaunchKernel | 9495 | 68,212 | 7.18 | 2.8 |
| cudaFree | 85 | 46,358 | 545.39 | 1.9 |
| cudaHostAlloc | 7 | 7236.3 | 1033.76 | 0.3 |
| cudaLaunchKernelExC_v11060 | 644 | 2439.6 | 3.79 | 0.1 |
| cudaDeviceSynchronize | 33 | 2298.1 | 69.64 | 0.1 |
| cudaFreeHost | 7 | 2003.0 | 286.14 | 0.1 |
| cuKernelGetName | 9495 | 647.1 | 0.07 | 0.0 |
| cudaEventRecord | 126 | 340.2 | 2.70 | 0.0 |
| cudaMemsetAsync | 119 | 338.8 | 2.85 | 0.0 |
| cudaMemcpyAsync | 56 | 304.3 | 5.43 | 0.0 |
