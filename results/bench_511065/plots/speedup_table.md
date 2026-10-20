GPU: NVIDIA H200. time_ms = median of reps; speedup = native_loop time / path time.

| workload | native_loop ms | native_grouped ms | emu_loop ms | emu_grouped ms | auto ms | emu_grouped_reuse ms | emu_loop_batched ms | native_grouped x | emu_loop x | emu_grouped x | auto x | emu_grouped_reuse x | emu_loop_batched x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| het_G256_16-2048 | 3.0708 | 1.9709 | 72.1637 | 9.7146 | 2.0079 | 10.6656 | 37.0613 | 1.56 | 0.04 | 0.32 | 1.53 | 0.29 | 0.08 |
| lora_G64_rmix_expand | 0.4445 | 0.1176 | 9.0700 | 2.1440 | 0.1214 | 2.8365 | 3.9840 | 3.78 | 0.05 | 0.21 | 3.66 | 0.16 | 0.11 |
| lora_G64_rmix_shrink | 0.8442 | 0.1658 | 93.8520 | 2.1083 | 0.1723 | 2.0925 | 56.7866 | 5.09 | 0.01 | 0.40 | 4.90 | 0.40 | 0.01 |
| moe_E64_prefill_z1.2_down | 1.5966 | 1.5887 | 42.8143 | 6.8264 | 1.5970 | 6.9449 | 26.8510 | 1.00 | 0.04 | 0.23 | 1.00 | 0.23 | 0.06 |
| moe_E64_prefill_z1.2_up | 1.6544 | 1.5684 | 57.3313 | 6.7115 | 1.5762 | 6.6035 | 36.3008 | 1.05 | 0.03 | 0.25 | 1.05 | 0.25 | 0.05 |
| uni_G1024_s32 | 7.2653 | 0.0735 | 116.4347 | 1.2533 | 0.1855 | 1.3966 | 37.3012 | 98.84 | 0.06 | 5.80 | 39.16 | 5.20 | 0.19 |
