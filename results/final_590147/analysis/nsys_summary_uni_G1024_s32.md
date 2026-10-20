# nsys summary: uni_G1024_s32
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 1024 | 0 | 10,991 | 2092.9 | 8899.0 | 81.0 | 3368.8 | 0.0 | 11,017 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 32.4 | 23.6 | 8.9 | 27.6 | 13.4 | 0.0 | 54.3 | 0.0 | 0.0 | 0.0 | 89.8 | 10.2 | 0.0 |
| emu_loop | 5 | 18,432 | 1 | 112,060 | 94,280 | 17,785 | 15.9 | 77,718 | 6837.7 | 112,728 | 13.4 | 71.5 | 14.9 | 0.0 | 0.1 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 604.4 | 599.0 | 5.3 | 0.9 | 21.4 | 707.8 | 733.9 | 34.0 | 49.9 | 9.1 | 0.0 | 7.1 | 0.0 |
| auto | 5 | 2 | 1 | 32.1 | 23.5 | 8.5 | 26.5 | 12.8 | 152.0 | 171.0 | 0.0 | 0.0 | 0.0 | 89.7 | 10.3 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 701.8 | 694.9 | 6.9 | 1.0 | 22.7 | 748.7 | 777.5 | 43.1 | 42.8 | 7.9 | 0.0 | 6.2 | 0.0 |
| emu_loop_batched | 5 | 3072 | 1 | 35,423 | 32,394 | 3025.7 | 8.5 | 19,376 | 824.4 | 24,483 | 39.1 | 16.3 | 44.4 | 0.0 | 0.2 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,408,960 | 471.07 | 57.3 |
| cudaLaunchKernel | 150,587 | 683,695 | 4.54 | 27.8 |
| cudaStreamCreateWithFlags | 1 | 159,622 | 159,622 | 6.5 |
| cudaEventSynchronize | 77 | 96,624 | 1254.86 | 3.9 |
| cudaHostAlloc | 7 | 32,164 | 4594.92 | 1.3 |
| cudaLaunchKernelExC_v11060 | 7168 | 23,917 | 3.34 | 1.0 |
| cudaStreamSynchronize | 21 | 19,073 | 908.23 | 0.8 |
| cuKernelGetName | 150,587 | 9701.4 | 0.06 | 0.4 |
| cudaFreeHost | 7 | 9448.8 | 1349.83 | 0.4 |
| cudaFree | 85 | 8220.1 | 96.71 | 0.3 |
| cudaMalloc | 77 | 7199.6 | 93.50 | 0.3 |
| cudaMemcpyAsync | 56 | 882.0 | 15.75 | 0.0 |
| cudaEventRecord | 126 | 331.2 | 2.63 | 0.0 |
| cuGetProcAddress_v2 | 880 | 210.4 | 0.24 | 0.0 |
| cudaDeviceSynchronize | 33 | 140.4 | 4.25 | 0.0 |
