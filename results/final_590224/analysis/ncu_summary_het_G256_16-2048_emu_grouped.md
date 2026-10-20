# ncu summary: het_G256_16-2048_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 4166.4 | 23.3 | 24.0 | 35.3 | 25.0 | 0.0* | 128 | 49,280 | 2/0/8 | wait 23%, long_scoreboard 23%, short_scoreboard 13% |
| prep | `oz2_prep_kernel` | 2 | 5350.6 | 0.0 | 11.6 | 47.4 | 36.1 | 37.5 | 72 | 28,416 | 3/4/8 | short_scoreboard 33%, wait 24%, mio_throttle 10% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 2756.2 | 0.0 | 25.1 | 64.8 | 47.4 | 50.0 | 64 | 0 | 4/32/8 | long_scoreboard 47%, wait 22%, not_selected 14% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
