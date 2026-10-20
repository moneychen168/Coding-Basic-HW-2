# nsys summary: nsys_lora_G64_rmix_shrink
Call detection: NVTX ranges ggemm:<path>; every value is the median over the timed calls of that path. Stage % = share of GPU busy time.

| path | timed calls | kernels/call | memops/call | GPU span (us) | GPU busy (us) | launch gaps (us) | gap % of span | host API (us) | host plan (us) | host total (us) | prep (scale+residue) % | INT8 grouped GEMM % | reconstruct (CRT) % | cuBLAS FP64 % | memcpy/memset % | other kernels % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native_loop | 5 | 128 | 0 | 890.0 | 669.2 | 219.8 | 24.7 | 523.5 | 0.0 | 831.1 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| native_grouped | 5 | 2 | 1 | 169.0 | 154.0 | 15.0 | 8.9 | 17.9 | 0.0 | 26.7 | 0.0 | 0.0 | 0.0 | 99.4 | 0.6 | 0.0 |
| emu_loop | 5 | 1152 | 1 | 93,710 | 92,392 | 1321.3 | 1.4 | 10,073 | 420.8 | 11,892 | 57.9 | 41.5 | 0.6 | 0.0 | 0.0 | 0.0 |
| emu_grouped | 5 | 3 | 1 | 2046.6 | 2041.3 | 5.4 | 0.3 | 16.9 | 38.7 | 59.7 | 86.5 | 12.7 | 0.5 | 0.0 | 0.3 | 0.0 |
| auto | 5 | 2 | 1 | 166.9 | 153.9 | 12.5 | 7.5 | 13.3 | 10.0 | 29.1 | 0.0 | 0.0 | 0.0 | 99.4 | 0.6 | 0.0 |
| emu_grouped_reuse | 5 | 3 | 1 | 2339.8 | 2334.7 | 5.4 | 0.2 | 17.6 | 42.9 | 63.2 | 88.2 | 11.1 | 0.4 | 0.0 | 0.3 | 0.0 |
| emu_loop_batched | 5 | 192 | 1 | 56,818 | 56,554 | 262.9 | 0.5 | 663.4 | 64.4 | 956.0 | 94.6 | 4.4 | 1.0 | 0.0 | 0.0 | 0.0 |

## CUDA API summary (whole process, incl. setup)

| API | calls | total (us) | avg (us) | % of API time |
| --- | ---: | ---: | ---: | ---: |
| cuLibraryLoadData | 2991 | 1,489,608 | 498.03 | 53.8 |
| cudaEventSynchronize | 77 | 714,500 | 9279.22 | 25.8 |
| cudaStreamSynchronize | 21 | 202,777 | 9656.06 | 7.3 |
| cudaStreamCreateWithFlags | 1 | 168,764 | 168,764 | 6.1 |
| cudaLaunchKernel | 9880 | 157,940 | 15.99 | 5.7 |
| cudaMalloc | 77 | 12,483 | 162.11 | 0.5 |
| cudaFree | 85 | 9117.8 | 107.27 | 0.3 |
| cudaHostAlloc | 7 | 5338.6 | 762.66 | 0.2 |
| cudaFreeHost | 7 | 1960.2 | 280.02 | 0.1 |
| cuLaunchKernel | 413 | 1469.7 | 3.56 | 0.1 |
| cuKernelGetName | 9880 | 685.7 | 0.07 | 0.0 |
| cudaEventRecord | 126 | 480.2 | 3.81 | 0.0 |
| cudaMemcpyAsync | 56 | 410.7 | 7.33 | 0.0 |
| cudaLaunchKernelExC_v11060 | 70 | 317.0 | 4.53 | 0.0 |
| cudaDeviceSynchronize | 33 | 271.4 | 8.22 | 0.0 |
