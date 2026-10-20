# Roofline: bench

GPU NVIDIA H200: FP64 peak 61.33 TFLOPS, INT8 peak 1676 TOPS, DRAM 3.94 TB/s (calib JSON). Roof of a path = FP64 roof (native) or INT8 peak / N (emulated) capped by DRAM bandwidth x AI.

| workload | path | N | AI (flop/B) | GFLOPS | roof GFLOPS | % of roof | bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| het_G256_16-2048 | native_loop | - | 36.06 | 13,897 | 61,326 | 22.7 | compute |
| het_G256_16-2048 | native_grouped | - | 36.06 | 21,652 | 61,326 | 35.3 | compute |
| het_G256_16-2048 | emu_loop | 15 | 5.11 | 591.3 | 20,174 | 2.9 | memory |
| het_G256_16-2048 | emu_grouped | 15 | 5.11 | 4392.7 | 20,174 | 21.8 | memory |
| het_G256_16-2048 | emu_grouped_reuse | 15 | 5.11 | 4001.0 | 20,174 | 19.8 | memory |
| het_G256_16-2048 | emu_loop_batched | 15 | 5.11 | 1151.4 | 20,174 | 5.7 | memory |
| het_G256_16-2048 | auto (all fallback) | - | 36.06 | 21,253 | 61,326 | 34.7 | compute |
| lora_G64_rmix_expand | native_loop | - | 3.15 | 972.2 | 12,444 | 7.8 | memory |
| lora_G64_rmix_expand | native_grouped | - | 3.15 | 3674.7 | 12,444 | 29.5 | memory |
| lora_G64_rmix_expand | emu_loop | 15 | 0.33 | 47.6 | 1299 | 3.7 | memory |
| lora_G64_rmix_expand | emu_grouped | 15 | 0.33 | 201.6 | 1299 | 15.5 | memory |
| lora_G64_rmix_expand | emu_grouped_reuse | 15 | 0.33 | 152.4 | 1299 | 11.7 | memory |
| lora_G64_rmix_expand | emu_loop_batched | 15 | 0.33 | 108.5 | 1299 | 8.3 | memory |
| lora_G64_rmix_expand | auto (all fallback) | - | 3.15 | 3559.0 | 12,444 | 28.6 | memory |
| lora_G64_rmix_shrink | native_loop | - | 3.15 | 511.9 | 12,444 | 4.1 | memory |
| lora_G64_rmix_shrink | native_grouped | - | 3.15 | 2607.0 | 12,444 | 21.0 | memory |
| lora_G64_rmix_shrink | emu_loop | 15 | 1.08 | 4.6 | 4268 | 0.1 | memory |
| lora_G64_rmix_shrink | emu_grouped | 15 | 1.08 | 205.0 | 4268 | 4.8 | memory |
| lora_G64_rmix_shrink | emu_grouped_reuse | 15 | 1.08 | 206.5 | 4268 | 4.8 | memory |
| lora_G64_rmix_shrink | emu_loop_batched | 15 | 1.08 | 7.6 | 4268 | 0.2 | memory |
| lora_G64_rmix_shrink | auto (all fallback) | - | 3.15 | 2508.3 | 12,444 | 20.2 | memory |
| moe_E64_prefill_z1.2_down | native_loop | - | 21.52 | 22,193 | 61,326 | 36.2 | compute |
| moe_E64_prefill_z1.2_down | native_grouped | - | 21.52 | 22,303 | 61,326 | 36.4 | compute |
| moe_E64_prefill_z1.2_down | emu_loop | 15 | 5.85 | 827.6 | 23,089 | 3.6 | memory |
| moe_E64_prefill_z1.2_down | emu_grouped | 15 | 5.85 | 5190.7 | 23,089 | 22.5 | memory |
| moe_E64_prefill_z1.2_down | emu_grouped_reuse | 15 | 5.85 | 5102.1 | 23,089 | 22.1 | memory |
| moe_E64_prefill_z1.2_down | emu_loop_batched | 15 | 5.85 | 1319.6 | 23,089 | 5.7 | memory |
| moe_E64_prefill_z1.2_down | auto (all fallback) | - | 21.52 | 22,188 | 61,326 | 36.2 | compute |
| moe_E64_prefill_z1.2_up | native_loop | - | 21.52 | 21,417 | 61,326 | 34.9 | compute |
| moe_E64_prefill_z1.2_up | native_grouped | - | 21.52 | 22,593 | 61,326 | 36.8 | compute |
| moe_E64_prefill_z1.2_up | emu_loop | 15 | 6.28 | 618.0 | 24,779 | 2.5 | memory |
| moe_E64_prefill_z1.2_up | emu_grouped | 15 | 6.28 | 5279.6 | 24,779 | 21.3 | memory |
| moe_E64_prefill_z1.2_up | emu_grouped_reuse | 15 | 6.28 | 5365.8 | 24,779 | 21.7 | memory |
| moe_E64_prefill_z1.2_up | emu_loop_batched | 15 | 6.28 | 976.1 | 24,779 | 3.9 | memory |
| moe_E64_prefill_z1.2_up | auto (all fallback) | - | 21.52 | 22,481 | 61,326 | 36.7 | compute |
| uni_G1024_s32 | native_loop | - | 2.67 | 9.2 | 10,520 | 0.1 | memory |
| uni_G1024_s32 | native_grouped | - | 2.67 | 913.0 | 10,520 | 8.7 | memory |
| uni_G1024_s32 | emu_loop | 15 | 0.37 | 0.6 | 1451 | 0.0 | memory |
| uni_G1024_s32 | emu_grouped | 15 | 0.37 | 53.5 | 1451 | 3.7 | memory |
| uni_G1024_s32 | emu_grouped_reuse | 15 | 0.37 | 48.1 | 1451 | 3.3 | memory |
| uni_G1024_s32 | emu_loop_batched | 15 | 0.37 | 1.8 | 1451 | 0.1 | memory |
| uni_G1024_s32 | auto (all fallback) | - | 2.67 | 361.7 | 10,520 | 3.4 | memory |
