# 10/7 討論報告（書面版）

> 本檔把 10/07 實際報告的內容整理成書面版，內容逐字取自當天影片 `docs/video/report_1007.mp4` 的投影片與旁白講稿（`docs/video/slides.json`），沒有加入 10/07 之後才有的資訊。
> 數字是 10/07 當時的實測（job id 見各節「來源」），之後已被最終版本取代：最終結果見 `docs/report.md`，哪些 10/07 數字已過時見 `docs/report.md` §10.5b。會前大綱見 `docs/discussion_1007.md`。

討論報告要求三部分：初步結果、目前問題、最後一週計畫。

## 0. 題目與方法

### 投影片 1：Grouped GEMM 延伸題：最新進度

- Coding-Basic-HW-2 延伸題：Batched/Grouped GEMM
- Ozaki scheme II（CRT）+ INT8 Tensor Core
- 以 INT8 模擬 FP64，並做成 grouped 版本
- 進度報告 10/07（更新版）

**講稿：** 大家好，這是 Coding-Basic-HW-2 延伸題，Batched 與 Grouped GEMM。我們用 Ozaki scheme II，以 CRT 搭配 INT8 Tensor Core 模擬 FP64 矩陣乘法，並做成 grouped 版本。負面結果照實說明。

### 投影片 2：方法：Ozaki scheme II

- FP64 以 2 的冪次 scale 截斷成整數
- 對 N 個兩兩互質的模數（≤256）取 int8 residue
- INT8 GEMM，INT32 精確累加
- CRT 重建，約 67 bit 定點
- 最後反 scale 回 FP64

**講稿：** FP64 輸入先以二的冪次 scale 截成整數，再對 N 個兩兩互質、不超過 256 的模數取 int8 residue。每個模數做一次 INT8 GEMM，以 INT32 精確累加，再用 CRT 重建約 67 bit 的定點結果，最後反 scale 回 FP64。

### 投影片 3：Grouped 設計與新的 baseline

- EMU_LOOP：每題 N+2 次 launch，naive 下界
- EMU_LOOP_BATCHED：仿 GEMMul8 strided-batched，每題 3 次 launch
- EMU_GROUPED：整組固定 3 個 kernel
- prep → persistent grouped INT8 GEMM → reconstruct
- 主要加速比：EMU_LOOP_BATCHED → EMU_GROUPED

**講稿：** Grouped 版只用三個 kernel，prep、persistent grouped INT8 GEMM、reconstruct。比較基準方面，naive 的 EMU_LOOP 每題要 N 加 2 次 launch；新 baseline EMU_LOOP_BATCHED 仿 GEMMul8 strided-batched，每題三次 launch，加速比以它為準。

## 1. 初步結果

### 投影片 4：平台與校正實測

- nano4（國網），NVIDIA H200，CUDA 13.0
- launch 約 2.07 µs
- DRAM 3.94 TB/s
- FP64 GEMM 擬合峰值 61.3 TFLOPS
- INT8 GEMM 擬合峰值 1,676 TOPS

來源：來源：bench/calib/params_nvidia_h200.json（calib_510850.csv）

**講稿：** 平台是國網 nano4 的 H200，CUDA 13.0。實測launch 開銷約 2.07 微秒，DRAM 頻寬約 3.94 TB/s，FP64 GEMM 擬合峰值 61.3 TFLOPS，INT8 GEMM 擬合峰值 1,676 TOPS。這些參數餵給 AUTO planner 的 cost model。

### 投影片 5：正確性：certify-and-repair

- abs plane 下界 + 逐 entry 嚴格誤差界做認證
- 未通過的 entry：FP64 repair
- phi=2 N=15：COUNT 未認證 899；ON 修復 899，gate 5.919→0.287
- OFF 與 COUNT bitwise 相同；numerics 全數 PASSED
- 規則處理：EMU 系列預設 COUNT（計時路徑無 FP64 dot），只有 AUTO 用 ON

來源：job 511009（numerics log）

**講稿：** 正確性改用 certify-and-repair：逐 entry 以嚴格誤差界認證，未通過的用 FP64 repair。以 phi 等於 2、N 等於 15 為例，OFF 與 COUNT 的 gate 都是 5.919，COUNT 找出 899 個未認證 entry，ON 修復後 gate 降到 0.287，OFF 與 COUNT 逐位元相同。EMU 系列預設 COUNT，只有 AUTO 用 ON，計時中的 emulation 不做 FP64 dot。

### 投影片 6：測試結果（commit da18425）

- 完整正確性測試：2955 項通過、0 項失敗
- EMU 系列用 COUNT：certificate 不可漏報；AUTO 用 ON：全部過 γ_n 門檻
- CUDA Graph plan/execute：63/63，與直接呼叫 bit-exact
- operand reuse：30/30（EXACT_N 與 PROMOTE）
- planner 單元測試 1607/1607；數值單元測試 PASSED

來源：job 511059（tests）、511060（graph/reuse/num）、511063（planner）

**講稿：** 在 commit da18425 上重跑全部測試，完整正確性測試 2955 項全部通過，零失敗。純 emulation 用 COUNT，超標元素必須被 certificate 抓到；AUTO 用 ON，全部落在 FP64 誤差界內。CUDA Graph 六十三項、operand reuse 三十項、planner 一千六百零七項與數值測試也全數通過。

### 投影片 7：效能：EMU_LOOP_BATCHED → EMU_GROUPED

| 案例（H200） | LOOP_BATCHED | GROUPED | 加速 | AUTO→native |
|---|---|---|---|---|
| LoRA G=16 | 13.89 ms | 1.66 ms | 8.4× | 0.22 ms |
| TP G=8 | 8.15 ms | 1.58 ms | 5.2× | 0.37 ms |
| none G=16 | 4.16 ms | 0.65 ms | 6.4× | 0.16 ms |
| uniform G=1024（vs naive） | 107.8 ms | 1.04 ms | 104× | 0.18 ms |

來源：形狀 16×256×4096、256×512×4096、256×256×1024；job 510998/510894

**講稿：** 相對 EMU_LOOP_BATCHED，EMU_GROUPED 讓LoRA 十六題從 13.9 毫秒降到 1.66 毫秒，快 8.4 倍；TP 八題從 8.15 降到 1.58 毫秒，快 5.2 倍；不共用的十六題從 4.16 降到 0.65 毫秒，快 6.4 倍。對照 naive 的 EMU_LOOP，uniform 一千零二十四題從 107.8 毫秒降到 1.04 毫秒。AUTO 在 H200 上全改走 native FP64，只要 0.16 到 0.37 毫秒。

### 投影片 8：Operand reuse（path 5）

- 共用 A/B 只 prep 一次
- 與 EMU_GROUPED bit-exact（0 個 entry 不同）
- test_reuse_e2e：30/30 通過
- 加速 1.015–1.079×
- 加速有限，但沒有精度代價

來源：job 510998（test_reuse_e2e）

**講稿：** 第二項 operand reuse，即 path 5：共用的 A 或 B 只 prep 一次。與 EMU_GROUPED 完全 bit-exact，三十項測試全過。加速比約 1.02 到 1.08 倍，最高 1.079，最低 1.015。因為 prep 占比有限，但沒有精度代價。

## 2. 目前問題

### 投影片 9：INT8 kernel ablation：負面結果

- 第二輪 ablation 退步約 10%；revert_all 回不到 r1
- 結構性擾動，改回 r1 kernel
- r1：random 443、ozaki 544、8k 592、k16k 608 TOPS
- auto 439 TOPS；small256 auto 0.0237 ms
- split-K：同 kernel 時 cfg0 −3–5%；拆成第二個 kernel 時 small256 慢 31%、mix 慢 3% → 預設關閉

來源：job 510965（i8abl）；510989（i8unit）；511001、511054（split-K）

**講稿：** 第三項 INT8 kernel ablation 是負面結果：第二輪修改後整體退步約百分之十，連 revert all 都回不到 r1，故改回 r1。r1 實測 random 443、ozaki 544、8192 立方 592、k 等於 16384 是 608 TOPS；auto 為 439 TOPS，小問題 0.0237 毫秒。split-K 讓 cfg2 提升 37 到 53 百分比，但 cfg0 下降 3 到 5 百分比；拆成獨立 kernel 則因多一次 launch 與 tail 效應，小問題慢了 31%，所以預設關閉。

### 投影片 10：對 native 的誠實結論

- native：cuBLAS DgemmGroupedBatched
- H200 上 native FP64 很強，emulation 較慢
- 例：MoE up native 1.58 ms，emu_grouped 6.19 ms
- AUTO 的 cost model 全部 fallback native
- 價值：相對逐題 emulation 的 launch 合併

來源：job 510894（bench.csv，quick set）；較弱 FP64 硬體未實測

**講稿：** 最後，H200 native FP64 很強：cuBLAS DgemmGroupedBatched 做 MoE up 只要 1.58 毫秒，emu grouped 要 6.19 毫秒。AUTO 因此全部 fallback 到 native。在 FP64 很強的 GPU 上 emulation 不划算，grouped 的價值在於相對逐題 emulation 的 launch 合併；FP64 較弱的硬體我們沒有實測，不做宣稱。

## 3. 最後一週計畫

### 投影片 11：後續計畫與 AI 使用聲明

- 量測 repair 比例與成本、調 rho_budget
- 新 quick bench、端到端應用：結果待補
- profiling 證據：nsys launch gap、ncu tensor pipe
- 補完報告與 README，10/14 交件
- AI：Claude Code 多 agent 輔助；數字皆來自實測 job

**講稿：** 後續將補上新的 quick bench 與端到端應用結果，補 nsys 與 ncu 的 profiling 證據，完成報告與 README，十月十四日交件。AI 聲明：本專案以 Claude Code 多 agent 輔助實作，所有數字都來自實測 job。謝謝。
