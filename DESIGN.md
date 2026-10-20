# ggemm 設計與分工（內部契約）

題目：Coding-Basic-HW-2 延伸題「Batched/Grouped GEMM」。目標機器是 nano4（國網 H200，sm_90；使用者 10/07 確認）。程式仍同時編 sm_120。
nvcc：`/usr/local/cuda-13.0/bin/nvcc`，編 `-gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120`。

## 數值方法：Ozaki scheme II（CRT 版，INT8）
1. A 每一列、B 每一行各取 2 的冪次 scale，截斷成整數 A'、B'（截斷後的位元數依 K 與模數個數 N 決定，必須保證 |A'B'| < P/2，其中 P = ∏p_i）。
2. 對每個模數 p_i（≤256、兩兩互質）算出 residue (A' mod p_i)、(B' mod p_i)，以對稱範圍存成 int8。
3. INT8 GEMM，INT32 精確累加：C_i = A'_i B'_i（mod p_i）。kp ≤ 131040 才不會溢位（128²×131072=2³¹ 會溢位）。
4. 用 CRT 重建出 A'B'（double-double），反 scale 後套 alpha/beta，寫回 FP64 C。
參考資料：Ozaki/Uchino/Imamura 的 Ozaki scheme II 論文、GEMMul8（只能讀，不能抄碼，作業禁止提交既有實作）。

## Grouped 的優化重點（本作業的核心貢獻）
- Launch 合併：EMU_LOOP 每個問題要 O(N) 次 launch；EMU_GROUPED 整組固定是 3 個 kernel（prep → grouped INT8 GEMM → reconstruct），共用一次 descriptor H2D。
- 一個 persistent grouped INT8 GEMM 跑完所有 (problem × modulus) 的 tile；tile scheduler 依工作量排序（LPT），所有 grouped path 都使用。tile config 只有 AUTO 會依問題大小逐個選（128×128／64×64／32×32）；EMU_GROUPED／EMU_GROUPED_REUSE 固定用 128×128（cfg0）。
- Planner（AUTO）：每個問題各自決定模數個數 N（componentwise，`GGEMM_RHO_BUDGET`）；問題太小（TINY）、k ≤ 16（SMALL_K）、不划算或 K 太大（kp > 131040）時 fallback 到 native FP64，並記在 stats 裡。
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
- Kp = round_up(k, 32)，pad 補 0。k ≤ `OZ2_MAX_K` = 131040（INT32 累加不溢位）。
- **Planes**：num_moduli = N 的問題使用 N+1 份 plane（`oz2_num_planes(N)`），index l = 0..N。
  - l = 0..N−1：CRT residue（對稱 int8）。
  - l = N：**abs plane**，A'' = floor(2^σ_i |A|)、B'' = floor(2^τ_j |B|)，值為 0..127。它的 INT8 乘積是 (|A||B|) 的嚴格下界，供 componentwise 認證使用。
  - INT8 grouped GEMM 每個問題要跑 N+1 個 (prob, plane) 乘積，shape 全部相同；A8、B8、C32 都要配 N+1 份。stats 回報的 moduli 數仍然是 N。
- A residue：`int8 A8[plane][prob]`，每個問題是 row-major m × Kp（K 連續）。
- B residue：`int8 B8[plane][prob]`，每個問題是 n × Kp row-major（也就是 col-major 的 K × n，K 連續）。
- INT32 輸出：`int32 C32[plane][prob]`，每個問題是 col-major m × n（ld = m）。
- 每個 (prob, plane) 的 offset 由 P 在 host 端算好，放進 device descriptor 陣列。A8/B8 的 base 和 stride 要 4-byte 對齊。
- **Row/col workspace**：`row_shift` 每個問題要 4·m 個 int32，`col_shift` 要 4·n（`OZ2_ROW_WS_PER_ROW`）。內容是 {s_i | σ_i | ‖dA_i‖₂ 上界（float bits）| ‖A'_i‖₂ 上界（float bits）}。
- **Repair 模式**（`oz2_reconstruct_grouped` 的 `repair_mode` 參數，由整合端指定：EMU 系列用 COUNT，AUTO 用 ON）：
  - `OZ2_REPAIR_OFF`：純 emulation，不做認證。
  - `OZ2_REPAIR_COUNT`：計算嚴格誤差界 E_ij；無法證明 E_ij 滿足 componentwise γ_n 門檻的 entry 仍輸出 emulation 值，只計入 `uncertified`。
  - `OZ2_REPAIR_ON`：無法認證的 entry 在 kernel 內改用 FP64 FMA dot 重算，計入 `repaired`。
  - 各模式共通：A 列 / B 行含 NaN/Inf 的 entry 一律走 FP64 dot（safety fallback，計入 `safety_fallback`）；α = 0 時依 BLAS 語意輸出 β·C，且不讀 A/B（prep 會跳過這個問題，它的 plane 不會被寫入）。
  - 計數器用 `oz2_read_counters(oz2_counters*, reset, stream)` 讀取；它會同步 stream，只能在計時區外使用。

```c
// src/int8_grouped_gemm.h (agent-K)
struct i8_problem { int m, n, kp; const int8_t* A; const int8_t* B; int32_t* C; int tile_cfg; };
// d_probs: device 陣列；tile_cfg: 0=128x128, 1=64x64, 2=32x32（K 可再加）
int i8_grouped_gemm(const i8_problem* d_probs, int num, const i8_problem* h_probs_for_sched,
                    void* d_sched_ws, size_t sched_ws_bytes, cudaStream_t s);
size_t i8_grouped_gemm_sched_ws_bytes(int max_problems, size_t max_tiles);
```
```c
// src/ozaki2_numerics.h (agent-N) -- 完整說明見該檔
struct oz2_problem { int m, n, k, kp, num_moduli, lda, ldb, ldc;
                     const double *A, *B; double* C; double alpha, beta;
                     int8_t *A8, *B8;               // plane 0 的 base；plane 之間以 stride 隔開（共 N+1 份）
                     size_t a8_mod_stride, b8_mod_stride;
                     int32_t* C32; size_t c32_mod_stride;
                     int32_t *row_shift, *col_shift; };   // workspace：4*m、4*n 個 int32
int oz2_init_constants();                                    // 在 ggemm_create 呼叫
int oz2_prep_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s);   // 1 launch
int oz2_reconstruct_grouped(const oz2_problem* d, int num, int max_m, int max_n, cudaStream_t s,
                            int repair_mode = OZ2_REPAIR_ON);                                // 1 launch
int oz2_choose_num_moduli(int k, double target_rel_err);     // normwise 界
int oz2_choose_num_moduli_cw(int k, double rho_budget);      // componentwise 規劃（只影響 repair 比例）
double oz2_error_bound(int k, int num_moduli);
struct oz2_counters { unsigned long long nonfinite_vectors, uncertified, repaired, safety_fallback; };
int oz2_read_counters(oz2_counters* out, int reset, cudaStream_t s);
```

## 執行規則（全體遵守）
- 登入節點只能編譯，**不准執行任何 GPU/計算程式**（連 smoke test 都不行）。要跑的東西寫成 `scripts/` 底下的 SLURM 腳本，並把要跑的指令回報給主 agent，由主 agent 送 job。
- SLURM：`-A GOV114009`，`-p dev`，`--gpus-per-node=1`，`--cpus-per-task≤12`，`--time` 給寬鬆。
- 不准 git commit，由主 agent 統一 commit。
- 報告不能放推測的數字。
