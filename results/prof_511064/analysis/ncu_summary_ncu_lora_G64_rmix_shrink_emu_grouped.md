# ncu summary: ncu_lora_G64_rmix_shrink_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 335.8 | 48.0 | 17.3 | 47.1 | 24.0 | 0.0* | 128 | 49,280 | 2/0/8 | wait 30%, short_scoreboard 23%, not_selected 13% |
| prep | `oz2_prep_kernel` | 2 | 2055.0 | 0.0 | 5.3 | 18.5 | 27.4 | 62.5 | 48 | 28,416 | 5/5/8 | wait 36%, short_scoreboard 16%, long_scoreboard 14% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 13.7 | 0.0 | 5.3 | 19.2 | 30.8 | 50.0 | 64 | 0 | 4/32/8 | long_scoreboard 54%, wait 14%, short_scoreboard 11% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
