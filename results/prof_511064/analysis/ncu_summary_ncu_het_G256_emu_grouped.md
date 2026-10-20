# ncu summary: ncu_het_G256_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 4195.1 | 23.3 | 23.8 | 35.4 | 25.0 | 0.0* | 128 | 49,280 | 2/0/8 | wait 24%, long_scoreboard 23%, short_scoreboard 13% |
| prep | `oz2_prep_kernel` | 2 | 4788.3 | 0.0 | 13.6 | 51.9 | 58.5 | 62.5 | 48 | 28,416 | 5/5/8 | short_scoreboard 28%, wait 17%, mio_throttle 16% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 2744.1 | 0.0 | 25.2 | 59.8 | 47.5 | 50.0 | 64 | 0 | 4/32/8 | long_scoreboard 46%, wait 21%, not_selected 12% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
