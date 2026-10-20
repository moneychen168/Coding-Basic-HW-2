# ncu summary: uni_G1024_s32_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 370.0 | 4.9 | 4.5 | 39.7 | 24.9 | 0.0* | 128 | 49,280 | 2/0/8 | long_scoreboard 35%, wait 33%, barrier 8% |
| prep | `oz2_prep_kernel` | 2 | 267.3 | 0.0 | 2.2 | 58.2 | 36.2 | 37.5 | 72 | 28,416 | 3/4/8 | short_scoreboard 38%, wait 22%, mio_throttle 13% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 66.2 | 0.0 | 23.3 | 53.2 | 45.1 | 50.0 | 64 | 192 | 4/12/8 | long_scoreboard 50%, wait 18%, not_selected 13% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
