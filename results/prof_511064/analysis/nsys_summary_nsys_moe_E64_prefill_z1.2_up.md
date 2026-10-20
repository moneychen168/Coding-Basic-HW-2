# nsys summary: nsys_moe_E64_prefill_z1.2_up
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 117 | 0 | 1668.5 | 1493.8 | 175.4 | 10.5 | 384.1 | 0.0 | 855.1 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1578.4 | 1566.1 | 12.3 | 0.8 | 13.0 | 0.0 | 23.5 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 1152 | 1 | 57,431 | 55,789 | 1640.2 | 2.9 | 5552.5 | 337.4 | 7318.1 | 57.0 | 40.4 | 2.6 | 0.0 | 0.0 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 6671.1 | 6663.8 | 5.3 | 0.1 | 15.5 | 38.7 | 58.4 | 63.4 | 29.6 | 6.9 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1579.3 | 1567.1 | 11.9 | 0.8 | 11.5 | 11.4 | 29.9 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 6579.3 | 6573.8 | 6.9 | 0.1 | 15.1 | 43.0 | 61.7 | 62.9 | 29.9 | 7.0 | 0.0 | 0.1 | 0.0 |
| emu_loop_batched | 5 | 192 | 1 | 36,561 | 36,250 | 310.0 | 0.8 | 553.6 | 48.7 | 819.7 | 87.9 | 8.3 | 3.8 | 0.0 | 0.0 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,383,026 | 462.40 | 57.1 |
| cudaEventSynchronize | 77 | 518,486 | 6733.58 | 21.4 |
| cudaStreamSynchronize | 21 | 160,981 | 7665.75 | 6.6 |
| cudaStreamCreateWithFlags | 1 | 157,504 | 157,504 | 6.5 |
| cudaLaunchKernel | 9495 | 92,821 | 9.78 | 3.8 |
| cudaFree | 85 | 49,746 | 585.25 | 2.1 |
| cudaMalloc | 77 | 43,434 | 564.08 | 1.8 |
| cudaHostAlloc | 7 | 7207.8 | 1029.68 | 0.3 |
| cudaLaunchKernelExC_v11060 | 644 | 2293.9 | 3.56 | 0.1 |
| cudaDeviceSynchronize | 33 | 2274.5 | 68.92 | 0.1 |
| cudaFreeHost | 7 | 1895.9 | 270.85 | 0.1 |
| cuKernelGetName | 9495 | 642.9 | 0.07 | 0.0 |
| cudaEventRecord | 126 | 323.2 | 2.56 | 0.0 |
| cudaMemsetAsync | 119 | 305.3 | 2.57 | 0.0 |
| cudaMemcpyAsync | 56 | 300.5 | 5.37 | 0.0 |
