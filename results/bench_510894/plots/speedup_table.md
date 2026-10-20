GPU: NVIDIA H200. time_ms = median of reps; speedup = native_loop time / path time.

| workload | native_loop ms | native_grouped ms | emu_loop ms | emu_grouped ms | auto ms | native_grouped x | emu_loop x | emu_grouped x | auto x |
|---|---|---|---|---|---|---|---|---|---|
| het_G256_16-2048 | 3.0406 | 1.9674 | 68.4736 | 9.1050 | 2.0111 | 1.55 | 0.04 | 0.33 | 1.51 |
| lora_G64_rmix_expand | 0.4471 | 0.1204 | 8.3886 | 1.9665 | 0.1295 | 3.71 | 0.05 | 0.23 | 3.45 |
| lora_G64_rmix_shrink | 0.8259 | 0.1656 | 88.8479 | 2.0323 | 0.1715 | 4.99 | 0.01 | 0.41 | 4.82 |
| moe_E64_prefill_z1.2_down | 1.5946 | 1.5872 | 40.4407 | 6.2769 | 1.5991 | 1.00 | 0.04 | 0.25 | 1.00 |
| moe_E64_prefill_z1.2_up | 1.6537 | 1.5768 | 54.5640 | 6.1906 | 1.5815 | 1.05 | 0.03 | 0.27 | 1.05 |
| uni_G1024_s32 | 7.3509 | 0.0703 | 107.8368 | 1.0362 | 0.1786 | 104.54 | 0.07 | 7.09 | 41.16 |
