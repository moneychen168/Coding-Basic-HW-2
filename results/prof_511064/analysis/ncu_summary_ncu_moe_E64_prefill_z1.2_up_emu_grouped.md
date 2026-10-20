# ncu summary: ncu_moe_E64_prefill_z1.2_up_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 2450.3 | 44.5 | 31.6 | 48.8 | 24.9 | 0.0* | 128 | 49,280 | 2/0/8 | wait 29%, short_scoreboard 18%, math_pipe_throttle 12% |
| prep | `oz2_prep_kernel` | 2 | 5521.0 | 0.0 | 23.6 | 70.0 | 60.6 | 62.5 | 48 | 28,416 | 5/5/8 | not_selected 29%, math_pipe_throttle 23%, wait 22% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 548.1 | 0.0 | 23.6 | 58.5 | 46.9 | 50.0 | 64 | 0 | 4/32/8 | long_scoreboard 46%, wait 21%, not_selected 12% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
