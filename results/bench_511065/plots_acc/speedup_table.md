GPU: NVIDIA H200. time_ms = median of reps; speedup = native_loop time / path time.

| workload | native_loop ms | emu_grouped ms | auto ms | emu_grouped x | auto x |
|---|---|---|---|---|---|
| het_G256_16-2048 | 3.0563 | 9.7887 | 2.0257 | 0.31 | 1.51 |
| lora_G64_rmix_expand | 0.4454 | 2.1411 | 0.1224 | 0.21 | 3.64 |
| lora_G64_rmix_shrink | 0.8350 | 2.0941 | 0.1736 | 0.40 | 4.81 |
| moe_E64_prefill_z1.2_down | 1.5911 | 7.2775 | 1.5976 | 0.22 | 1.00 |
| moe_E64_prefill_z1.2_up | 1.6523 | 7.2558 | 1.5865 | 0.23 | 1.04 |
| uni_G1024_s32 | 7.3142 | 1.2489 | 0.1893 | 5.86 | 38.64 |
