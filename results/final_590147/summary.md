# Final experiment matrix: job 590147

- host: 25a-hgpn004
- start: 2026-10-20T22:00:08+08:00  end: 2026-10-20T22:09:45+08:00
- git: b56631d7a6835973f788c962056eb319a490a650  dirty files: 9 (see provenance_git.txt, dirty.patch)
- GPU: NVIDIA H200, 580.65.06
- paths: native_loop native_grouped emu_loop emu_loop_batched emu_grouped emu_grouped_reuse auto 
- steps: 34 PASS, 0 FAIL, 0 SKIP

| step | status | rc | seconds | log |
|---|---|---:|---:|---|
| cuda_preflight | PASS | 0 | 0 | logs/cuda_preflight.log |
| provenance | PASS | 0 | 1 | logs/provenance.log |
| build | PASS | 0 | 44 | logs/build.log |
| build_moe_ffn_fp64 | PASS | 0 | 9 | logs/build_moe_ffn_fp64.log |
| build_block_sparse_fp64 | PASS | 0 | 4 | logs/build_block_sparse_fp64.log |
| test_i8_grouped | PASS | 0 | 15 | logs/test_i8_grouped.log |
| test_numerics_standalone | PASS | 0 | 2 | logs/test_numerics_standalone.log |
| test_grouped | PASS | 0 | 5 | logs/test_grouped.log |
| test_graph | PASS | 0 | 4 | logs/test_graph.log |
| test_reuse_e2e | PASS | 0 | 3 | logs/test_reuse_e2e.log |
| bench_grouped | PASS | 0 | 157 | logs/bench_grouped.log |
| app_moe_ffn | PASS | 0 | 10 | logs/app_moe_ffn.log |
| app_block_sparse | PASS | 0 | 128 | logs/app_block_sparse.log |
| nsys_uni_G1024_s32 | PASS | 0 | 21 | logs/nsys_uni_G1024_s32.log |
| nsys_moe_E64_prefill_z1.2_up | PASS | 0 | 8 | logs/nsys_moe_E64_prefill_z1.2_up.log |
| nsys_het_G256_16-2048 | PASS | 0 | 11 | logs/nsys_het_G256_16-2048.log |
| ncu_uni_G1024_s32_emu_grouped | PASS | 0 | 22 | logs/ncu_uni_G1024_s32_emu_grouped.log |
| ncu_uni_G1024_s32_auto | PASS | 0 | 14 | logs/ncu_uni_G1024_s32_auto.log |
| ncu_moe_E64_prefill_z1.2_up_emu_grouped | PASS | 0 | 28 | logs/ncu_moe_E64_prefill_z1.2_up_emu_grouped.log |
| ncu_moe_E64_prefill_z1.2_up_auto | PASS | 0 | 15 | logs/ncu_moe_E64_prefill_z1.2_up_auto.log |
| ncu_het_G256_16-2048_emu_grouped | PASS | 0 | 29 | logs/ncu_het_G256_16-2048_emu_grouped.log |
| ncu_het_G256_16-2048_auto | PASS | 0 | 15 | logs/ncu_het_G256_16-2048_auto.log |
| roofline | PASS | 0 | 3 | logs/roofline.log |
| nsys_report_uni_G1024_s32 | PASS | 0 | 8 | logs/nsys_report_uni_G1024_s32.log |
| ncu_report_uni_G1024_s32_emu_grouped | PASS | 0 | 1 | logs/ncu_report_uni_G1024_s32_emu_grouped.log |
| ncu_report_uni_G1024_s32_auto | PASS | 0 | 0 | logs/ncu_report_uni_G1024_s32_auto.log |
| nsys_report_moe_E64_prefill_z1.2_up | PASS | 0 | 1 | logs/nsys_report_moe_E64_prefill_z1.2_up.log |
| ncu_report_moe_E64_prefill_z1.2_up_emu_grouped | PASS | 0 | 1 | logs/ncu_report_moe_E64_prefill_z1.2_up_emu_grouped.log |
| ncu_report_moe_E64_prefill_z1.2_up_auto | PASS | 0 | 0 | logs/ncu_report_moe_E64_prefill_z1.2_up_auto.log |
| nsys_report_het_G256_16-2048 | PASS | 0 | 3 | logs/nsys_report_het_G256_16-2048.log |
| ncu_report_het_G256_16-2048_emu_grouped | PASS | 0 | 0 | logs/ncu_report_het_G256_16-2048_emu_grouped.log |
| ncu_report_het_G256_16-2048_auto | PASS | 0 | 0 | logs/ncu_report_het_G256_16-2048_auto.log |
| plots | PASS | 0 | 9 | logs/plots.log |
| aggregate | PASS | 0 | 0 | logs/aggregate.log |

Tables: tables.md (scripts/final/aggregate.py). Paths: paths.tsv.
