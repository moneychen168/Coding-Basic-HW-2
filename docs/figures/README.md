# docs/figures — 最終版 profiling 圖（job 590258）

產生方式：`scripts/analysis/plot_profiling.py results/final_590258 -o docs/figures`（只讀
`results/final_590258/analysis/` 底下既有的 CSV，不重新分析）。每張圖輸出 `.png`（200 dpi）與 `.svg`。

欄位定義（來自 `scripts/analysis/nsys_report.py`）：nsys 的 `*_ns` 都是**每次 call 的 median**
（`calls` = timed call 次數，不必再除）；`span_ns` = 第一個 GPU op 開始到最後一個 GPU op 結束；
`busy_ns` = span 內 GPU op 區間的聯集；`plan_ns` = NVTX `plan` range 內的 host 時間（CPU 規劃 /
descriptor 打包）；`*_pct` = 該 stage 佔 GPU busy 的百分比。

顏色固定對應實體（所有圖一致，取自 reference palette light-mode 類別色序）：
native_loop `#2a78d6`、native_grouped `#eb6834`、emu_loop `#1baf7a`、emu_grouped `#eda100`、
auto `#e87ba4`；stage：prep `#008300`、INT8 GEMM `#6250d6`、reconstruct `#e34948`、
other + memops `#9ca3af`；host plan（非路徑）`#9ca3af`。

| 圖檔 | 來源 CSV | 使用欄位 | 圖說（中文） |
|---|---|---|---|
| `fig_stage_breakdown` | `nsys_summary_<wl>.csv`（uni_G1024_s32、moe_E64_prefill_z1.2_up、het_G256_16-2048） | `path`（emu_loop、emu_grouped）、`prep_pct`、`i8_gemm_pct`、`reconstruct_pct`、`memops_pct` + `other_pct` + `cublas_pct`（合併為 other + memops） | 圖 1：模擬路徑每次 call 的 GPU busy 依 stage 拆分（100% 堆疊）。≥8% 的區段直接標示百分比；各列加總重新正規化為 100%。 |
| `fig_gpu_span` | `nsys_summary_<wl>.csv` | `path`（native_loop、native_grouped、emu_loop、emu_grouped）、`span_ns`（÷1e6 → ms） | 圖 2：每次 call 的 GPU span（ms，log 軸）。span 已是每次 call 的 median，**沒有**除以 `calls`。 |
| `fig_host_vs_gpu` | `nsys_summary_<wl>.csv` | `path`（emu_grouped、auto）、`plan_ns`、`busy_ns`（÷1e3 → µs） | 圖 3：host 規劃時間 vs GPU busy（µs，log 軸，每次 call median）。uni_G1024_s32 上 emu_grouped 的 host plan（618.9 µs）大於 GPU busy（595.7 µs）。auto 在這三個 workload 全部 fallback，GPU busy 是 cuBLAS。 |
| `fig_roofline` | `roofline_final.csv` | `workload`、`path`（native_loop、native_grouped、emu_loop、emu_grouped、auto (all fallback)）、`ai`、`gflops`、`roof_gflops`、`bound`、`pct_roof`、`N` | 圖 4：log-log roofline，只有 native_loop、emu_loop、emu_grouped 的點旁標示 workload 與該路徑自身 roof 的達成率（`pct_roof`）；native_grouped 與 auto（全部 fallback，點與 native_grouped 重疊）不標。x 軸範圍約 0.2–100。DRAM BW = memory-bound 列 `roof_gflops`×1e9/`ai` 的 median；FP64 峰值 = 非 emu 的 compute-bound 列 `roof_gflops` 的 median。emu 列全為 memory-bound，CSV 推不出 INT8 峰值/N，所以模擬的 compute 上限不畫，只畫延伸的 DRAM 斜線。 |
| `fig_ncu_kernels` | `ncu_summary_<wl>_emu_grouped.csv`（`ncu_summary_<wl>_auto.csv` 也會讀，但三個都只有表頭） | `stage`（prep、i8_gemm、reconstruct）、`imma_pct`、`dram_pct`、`sm_pct` | 圖 5：emu_grouped 三個 kernel 的 ncu 利用率（小倍數，每個 workload 一格，y 軸統一 0–100%）。imma_pct = tensor pipe（IMMA）active %、dram_pct = `dram__throughput` % of peak、sm_pct = `sm__throughput` % of peak。AUTO 全部 fallback，沒有模擬 kernel 列。 |
