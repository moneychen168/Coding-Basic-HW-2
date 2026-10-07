# ggemm 設計與分工（內部契約）

題目：Coding-Basic-HW-2 延伸題「Batched/Grouped GEMM」。目標機器是 nano4 RTX 5090（sm_120，FP64 很弱）；開發機是國網 H100/H200（sm_90）。
nvcc：`/usr/local/cuda-13.0/bin/nvcc`，編 `-gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120`。

## 數值方法：Ozaki scheme II（CRT 版，INT8）
1. A 每一列、B 每一行各取 2 的冪次 scale，截斷成整數 A'、B'（截斷後的位元數依 K 與模數個數 N 決定，必須保證 |A'B'| < P/2，其中 P = ∏p_i）。
2. 對每個模數 p_i（≤256、兩兩互質）算出 residue (A' mod p_i)、(B' mod p_i)，以對稱範圍存成 int8。
3. INT8 GEMM，INT32 精確累加：C_i = A'_i B'_i（mod p_i）。kp ≤ 131040 才不會溢位（128²×131072=2³¹ 會溢位）。
4. 用 CRT 重建出 A'B'（double-double），反 scale 後套 alpha/beta，寫回 FP64 C。
參考資料：Ozaki/Uchino/Imamura 的 Ozaki scheme II 論文、GEMMul8（只能讀，不能抄碼，作業禁止提交既有實作）。

## Grouped 的優化重點（本作業的核心貢獻）
- Launch 合併：EMU_LOOP 每個問題要 O(N) 次 launch；EMU_GROUPED 整組固定是 3 個 kernel（prep → grouped INT8 GEMM → reconstruct），共用一次 descriptor H2D。
- 一個 persistent grouped INT8 GEMM 跑完所有 (problem × modulus) 的 tile；tile scheduler 依工作量排序（LPT）。tile config 依問題大小逐個選。
- Planner（AUTO）：每個問題各自決定模數個數 N；問題太小、不划算或 K 太大時 fallback 到 native FP64，並記在 stats 裡。
- 計時區內不做任何 allocation。所有 workspace 都在 ggemm_create 時配好，包括 pinned host descriptor buffer。

## 模組與檔案歸屬（每個 agent 只改自己的檔案）
| 模組 | 檔案 | 負責 |
|---|---|---|
| 公開 API | include/ggemm.h | 主 agent（要改請提出） |
| 內部契約 | src/internal.h | 主 agent 起稿，P 負責維護；N/K 只能新增自己的宣告 |
| 數值 N | src/ozaki2_numerics.cu/.h, src/moduli.h | agent-N |
| INT8 grouped GEMM K | src/int8_grouped_gemm.cu/.h | agent-K |
| Pipeline/Planner P | src/ggemm.cu, src/planner.cpp/.h, CMakeLists.txt | agent-P |
| Bench/Test B | tests/*, bench/*, scripts/* | agent-B |
| 文獻 | docs/related_work.md | agent-L |
| 10/7 討論簡報大綱 | docs/discussion_1007.md | agent-S |

## 內部資料佈局（N、K、P 共同遵守）
- Kp = round_up(k, 32)，pad 補 0。
- A residue：`int8 A8[mod][prob]`，每個問題是 row-major m × Kp（K 連續）。
- B residue：`int8 B8[mod][prob]`，每個問題是 n × Kp row-major（也就是 col-major 的 K × n，K 連續）。
- INT32 輸出：`int32 C32[mod][prob]`，每個問題是 col-major m × n（ld = m）。
- 每個 (prob, mod) 的 offset 由 P 在 host 端算好，放進 device descriptor 陣列。

```c
// src/int8_grouped_gemm.h (agent-K)
struct i8_problem { int m, n, kp; const int8_t* A; const int8_t* B; int32_t* C; int tile_cfg; };
// d_probs: device 陣列；tile_cfg: 0=128x128, 1=64x64, 2=32x32（K 可再加）
int i8_grouped_gemm(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                    void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s);
size_t i8_grouped_gemm_sched_ws_bytes(int max_problems, size_t max_tiles);
```
```c
// src/ozaki2_numerics.h (agent-N)
struct oz2_problem { int m, n, k, kp, num_moduli, lda, ldb, ldc;
                     const double *A, *B; double* C; double alpha, beta;
                     int8_t *A8, *B8;               // mod 0 的 base；同一問題的模數之間以 stride 隔開
                     size_t a8_mod_stride, b8_mod_stride;
                     int32_t* C32; size_t c32_mod_stride;
                     int32_t *row_shift, *col_shift; };   // workspace
int oz2_prep_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s);   // A、B 的 scale + residue（全部問題一次 launch）
int oz2_reconstruct_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s);
int oz2_choose_num_moduli(int k, double target_rel_err);   // host 端
```

## 執行規則（全體遵守）
- 登入節點只能編譯，**不准執行任何 GPU/計算程式**（連 smoke test 都不行）。要跑的東西寫成 `scripts/` 底下的 SLURM 腳本，並把要跑的指令回報給主 agent，由主 agent 送 job。
- SLURM：`-A GOV114009`，`-p dev`，`--gpus-per-node=1`，`--cpus-per-task≤12`，`--time` 給寬鬆。
- 不准 git commit，由主 agent 統一 commit。
- 報告不能放推測的數字。
