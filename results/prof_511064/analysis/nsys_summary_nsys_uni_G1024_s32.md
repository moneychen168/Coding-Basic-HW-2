# nsys summary: nsys_uni_G1024_s32
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 1024 | 0 | 10,720 | 2072.0 | 8646.9 | 80.7 | 3118.2 | 0.0 | 10,731 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 31.7 | 23.4 | 8.4 | 26.3 | 12.2 | 0.0 | 58.3 | 0.0 | 0.0 | 0.0 | 89.9 | 10.1 | 0.0 |
| emu_loop | 5 | 18,432 | 1 | 113,365 | 95,584 | 17,776 | 15.7 | 83,762 | 5794.7 | 112,905 | 12.6 | 71.5 | 15.8 | 0.0 | 0.1 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 577.5 | 569.3 | 6.0 | 1.0 | 21.5 | 675.7 | 702.8 | 30.4 | 52.8 | 10.1 | 0.0 | 6.7 | 0.0 |
| auto | 5 | 2 | 1 | 31.9 | 23.4 | 8.5 | 26.8 | 12.1 | 152.7 | 171.5 | 0.0 | 0.0 | 0.0 | 90.0 | 10.0 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 673.7 | 665.6 | 9.9 | 1.5 | 23.6 | 743.7 | 772.6 | 40.3 | 44.9 | 8.6 | 0.0 | 6.3 | 0.0 |
| emu_loop_batched | 5 | 3072 | 1 | 36,158 | 33,126 | 3033.9 | 8.4 | 20,735 | 800.6 | 24,977 | 36.3 | 16.2 | 47.3 | 0.0 | 0.2 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,423,300 | 475.86 | 56.5 |
| cudaLaunchKernel | 150,587 | 731,701 | 4.86 | 29.0 |
| cudaStreamCreateWithFlags | 1 | 162,543 | 162,543 | 6.4 |
| cudaEventSynchronize | 77 | 97,778 | 1269.84 | 3.9 |
| cudaHostAlloc | 7 | 28,261 | 4037.29 | 1.1 |
| cudaLaunchKernelExC_v11060 | 7168 | 22,183 | 3.09 | 0.9 |
| cudaStreamSynchronize | 21 | 19,610 | 933.82 | 0.8 |
| cuKernelGetName | 150,587 | 9900.1 | 0.07 | 0.4 |
| cudaFreeHost | 7 | 8778.6 | 1254.09 | 0.3 |
| cudaFree | 85 | 8637.5 | 101.62 | 0.3 |
| cudaMalloc | 77 | 7146.5 | 92.81 | 0.3 |
| cudaMemcpyAsync | 56 | 407.5 | 7.28 | 0.0 |
| cudaEventRecord | 126 | 267.0 | 2.12 | 0.0 |
| cuGetProcAddress_v2 | 880 | 172.2 | 0.20 | 0.0 |
| cudaDeviceSynchronize | 33 | 132.8 | 4.02 | 0.0 |
