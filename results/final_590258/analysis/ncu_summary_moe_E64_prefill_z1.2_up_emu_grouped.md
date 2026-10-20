# ncu summary: moe_E64_prefill_z1.2_up_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 2455.9 | 44.5 | 31.5 | 48.8 | 24.9 | 0.0* | 128 | 49,280 | 2/0/8 | wait 29%, short_scoreboard 18%, math_pipe_throttle 12% |
| prep | `oz2_prep_kernel` | 2 | 5571.4 | 0.0 | 22.7 | 66.8 | 36.8 | 37.5 | 72 | 28,416 | 3/4/8 | wait 35%, not_selected 19%, math_pipe_throttle 15% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 612.2 | 0.0 | 21.1 | 65.1 | 47.4 | 50.0 | 64 | 192 | 4/12/8 | long_scoreboard 41%, wait 22%, not_selected 14% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
