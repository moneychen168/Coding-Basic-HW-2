# ncu summary: ncu_uni_G1024_s32_auto

Median over launches of each kernel. INT8 tensor pipe metric: `sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active`. Stall source: cycles/issue (share of all stall reasons, `selected` excluded).

| stage | kernel | launches | duration (us) | INT8 tensor pipe % | DRAM thru % | SM thru % | achieved occ % | theor. occ % | regs/thread | smem/block (B) | block limit reg/smem/warp | top-3 stalls (% of stall) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cuBLAS | `Kernel2` | 2 | 29.4 | 0.0 | 12.0 | 37.6 | 12.2 | 12.5 | 190 | 65,552 | 2/2/16 | wait 38%, math_pipe_throttle 16%, short_scoreboard 15% |
