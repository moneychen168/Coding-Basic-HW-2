# ncu summary: ncu_uni_G1024_s32_emu_grouped

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| INT8 grouped GEMM | `i8_grouped_persistent_kernel` | 2 | 370.6 | 4.9 | 4.5 | 39.5 | 24.9 | 0.0* | 128 | 49,280 | 2/0/8 | long_scoreboard 35%, wait 33%, barrier 8% |
| prep | `oz2_prep_kernel` | 2 | 228.7 | 0.0 | 2.5 | 66.6 | 59.3 | 62.5 | 48 | 28,416 | 5/5/8 | short_scoreboard 31%, mio_throttle 20%, wait 16% |
| reconstruct | `oz2_reconstruct_kernel` | 2 | 70.5 | 0.0 | 22.0 | 50.6 | 44.6 | 50.0 | 64 | 0 | 4/32/8 | long_scoreboard 49%, wait 20%, not_selected 10% |

\* ncu itself reports theoretical occupancy 0 (and Block Limit Shared Mem 0) for this kernel while achieved occupancy is > 0, so that theoretical value is not usable; see the block-limit column.
