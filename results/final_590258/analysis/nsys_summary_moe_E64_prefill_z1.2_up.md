# nsys summary: moe_E64_prefill_z1.2_up
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 117 | 0 | 1680.0 | 1499.1 | 181.8 | 10.8 | 401.7 | 0.0 | 875.6 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 1585.2 | 1570.4 | 13.9 | 0.9 | 13.8 | 0.0 | 25.5 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_loop | 5 | 1152 | 1 | 43,321 | 41,625 | 1698.8 | 3.9 | 3848.2 | 453.8 | 5693.4 | 42.5 | 54.1 | 3.3 | 0.0 | 0.0 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 6801.9 | 6793.6 | 7.6 | 0.1 | 15.7 | 45.1 | 63.9 | 63.4 | 29.0 | 7.5 | 0.0 | 0.1 | 0.0 |
| auto | 5 | 2 | 1 | 1579.8 | 1569.1 | 10.8 | 0.7 | 11.3 | 9.7 | 27.2 | 0.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 6704.2 | 6696.9 | 7.3 | 0.1 | 15.9 | 46.9 | 66.7 | 63.0 | 29.2 | 7.6 | 0.0 | 0.1 | 0.0 |
| emu_loop_batched | 5 | 192 | 1 | 23,035 | 22,719 | 315.6 | 1.4 | 575.8 | 58.7 | 856.4 | 78.6 | 15.5 | 5.9 | 0.0 | 0.0 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,389,935 | 464.71 | 61.9 |
| cudaEventSynchronize | 77 | 389,828 | 5062.70 | 17.4 |
| cudaStreamCreateWithFlags | 1 | 164,415 | 164,415 | 7.3 |
| cudaStreamSynchronize | 21 | 121,706 | 5795.53 | 5.4 |
| cudaLaunchKernel | 9495 | 68,774 | 7.24 | 3.1 |
| cudaFree | 85 | 47,333 | 556.86 | 2.1 |
| cudaMalloc | 77 | 44,803 | 581.86 | 2.0 |
| cudaHostAlloc | 7 | 7310.0 | 1044.29 | 0.3 |
| cudaDeviceSynchronize | 33 | 4825.9 | 146.24 | 0.2 |
| cudaLaunchKernelExC_v11060 | 644 | 2403.1 | 3.73 | 0.1 |
| cudaFreeHost | 7 | 1981.4 | 283.06 | 0.1 |
| cuKernelGetName | 9495 | 647.7 | 0.07 | 0.0 |
| cudaEventRecord | 126 | 340.6 | 2.70 | 0.0 |
| cudaMemsetAsync | 119 | 339.2 | 2.85 | 0.0 |
| cudaMemcpyAsync | 56 | 316.2 | 5.65 | 0.0 |
