GPU: NVIDIA H200. time_ms = median of reps; speedup = native_loop time / path time.

| workload | native_loop ms | emu_grouped ms | auto ms | emu_grouped x | auto x |
|---|---|---|---|---|---|
| het_G256_16-2048 | 3.0392 | 9.0540 | 2.0199 | 0.34 | 1.50 |
| lora_G64_rmix_expand | 0.4444 | 1.9518 | 0.1300 | 0.23 | 3.42 |
| lora_G64_rmix_shrink | 0.8229 | 2.0169 | 0.1724 | 0.41 | 4.77 |
| moe_E64_prefill_z1.2_down | 1.5860 | 6.2653 | 1.6026 | 0.25 | 0.99 |
| moe_E64_prefill_z1.2_up | 1.6492 | 6.1961 | 1.5904 | 0.27 | 1.04 |
| uni_G1024_s32 | 7.3418 | 1.0392 | 0.1790 | 7.07 | 41.01 |
