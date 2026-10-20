# ggemm 期末報告：以 INT8 Tensor Core 模擬 FP64 的 Grouped GEMM（Ozaki scheme II）

> Coding-Basic-HW-2 延伸題「Batched/Grouped GEMM」。本報告的每個數字都取自 `results/` 或 `logs/` 底下的實測檔案，並在旁邊註明 SLURM job id（寫作「job NNNNNN」）和檔名。由檔內數字算出的比值會寫出算式。目前沒有待填的數字。**正式版本是 job 590258**（`results/final_590258/`，程式 commit 1c39e40，35 個步驟全部 PASS）。provenance 記錄 dirty=6，內容是 4 個修改過的文件（DESIGN.md、README.md、docs/discussion_1007.md、docs/related_work.md）和 2 個未追蹤項目（docs/report.md、scripts/analysis/__pycache__/），`src/` 是乾淨的。job 590147（b56631d）與 590224（a48eb3e）作為修正前後的對照（§9.6）。

---

## 0. 摘要與評分項目對照

**一句話結論。** 我們實作了完整的 grouped FP64 GEMM library `ggemm`，內容包括：Ozaki scheme II 的 CRT 流程、persistent grouped INT8 GEMM、逐 entry 的 certify-and-repair、planner、operand reuse，以及 CUDA Graph plan/execute。

**主要數字**（正式 job 590258，commit 1c39e40，`results/final_590258/tables.md`，35 個步驟全部 PASS）：

1. **模擬本身的進步**（純模擬之間的比較）：`EMU_GROUPED` 相對 `EMU_LOOP_BATCHED` 的 E2E speedup，44 個 workload 的 geomean 是 **4.97×**（uniform 18.49×、MoE 2.65×、hetero 2.23×、LoRA 3.25×；表 2c）。`EMU_LOOP_BATCHED` 相對 `EMU_LOOP` 是 1/0.41 ≈ 2.4×。
2. **對 native 的負面結果**：在本作業唯一可用的 GPU（國網 nano4 的 H200）上，`EMU_GROUPED` 在 **44/44** 個 workload 都比 `NATIVE_GROUPED`（cuBLAS）慢，慢 **2.38×–21.3×**（geomean 0.14×，表 2b）。
3. **AUTO 不是模擬的加速**：44 列中有 38 列 fallback = G。其餘 6 列（MoE decode）的 fallback 數等於非空問題數，剩下的是 m_e = 0 的空問題；44 列的 `moduli_max` 都是 0。也就是說，**AUTO 在 bench 中沒有任何一題用模擬計算**。AUTO 相對 native_grouped 的 geomean 是 1.03×（tables.md 標為「all-native」）。但這個 > 1 只是因為 6 列 MoE decode 的 native_grouped 被量測順序高估（§10.8），不代表 AUTO 比 native 快。
4. **兩項修正已經實測**（§9.6）：
   - planner 的 per-call shape memo 修掉了 AUTO 全部 fallback 時的開銷 regression。例如 uni_G4096_s32 從 0.738 ms（590147）降到 0.2944 ms（590258），同列 native_grouped 是 0.2697 ms。
   - reconstruct 的計數器改成 warp + block 聚合後，repair sweep 的計數結果完全不變，時間 geomean 變成原來的 0.959 倍。
5. **native／starter／最終版三方比較**（job 590633，同一個 job、同一張 GPU、同樣 44 個 workload）：
   - 沒有官方 starter code，所以 starter = 我們第一個可用版本 v0（commit 5d8e2ad）。
   - v0 的 starter-like 路徑 EMU_LOOP 相對最終版 EMU_GROUPED 的時間比，geomean **13.9×**（範圍 2.04×–91.2×，§7.0）。
   - 最終版的 EMU_GROUPED 比 v0 的 EMU_GROUPED **慢 9%**（v0/final geomean 0.91×）。兩版用相同的 N、輸出逐位元相同；最終版多做了 abs plane 與逐 entry 認證，v0 沒有任何誤差認證。這 9% 是安全保證的代價（各部分的比例沒有分開量測）。
   - 兩版都遠慢於 native_grouped（最終版 native_grouped/emu_grouped 的 geomean 0.135×）。
6. **量測假象已經修正**：MoE decode 的 native_grouped 時間取決於它在同一個 process 裡是不是第一個被量。bench 矩陣中這幾列被高估約 1.6×，先前「AUTO 在 MoE decode 比 native 快」的說法是假象，已刪除（§10.8）。
7. **計時區沒有 host 配置**：590258 的 `bench.csv` 中 308 列的 `host_allocs_timed` 全部是 0。
8. **正確性**：`test_grouped` 2955 passed / 0 failed（job 590256、590258，commit 1c39e40）；compute-sanitizer 在最終程式碼上（job 590498，commit a6b337d，`src/` 與 1c39e40 相同）四種工具全部 0 errors。

§10 列出全部的負面結果。

| 評分要求 | 本報告章節 |
|---|---|
| 方法：Ozaki-II CRT pipeline、grouped persistent INT8 GEMM、prep/reconstruct、planner/AUTO、certify-and-repair、operand reuse、CUDA Graph | §2 |
| 實驗環境 | §3 |
| 測資與問題規模的選擇理由 | §4 |
| 正確性標準的定義與結果 | §5 |
| 在相同條件下比較 native、starter、最終版本 | §6（量測方法）、§7（結果） |
| E2E、GPU-busy、Graph replay、reuse 時間分開呈現 | §6.2、§7、§8.4 |
| profiling 證據與瓶頸分析 | §8 |
| Ablation | §9 |
| 失敗案例與負面結果 | §10 |
| AI 與外部程式碼使用說明 | §11 |
| 重現方式 | §12 |
| 數字來源總表 | 附錄 A |

---

## 1. 問題定義與符號

- **GEMM**（General Matrix Multiply）：C ← α·A·B + β·C。**FP64**：IEEE 754 雙精度浮點數。**INT8**／**INT32**：8 位元／32 位元有號整數。
- **Grouped GEMM**：一次呼叫處理 G 個互相獨立的問題，第 g 個問題是 C_g = α_g·A_g (m_g×k_g)·B_g (k_g×n_g) + β_g·C_g，每個問題的 shape、α、β、leading dimension 都可以不同。**Batched GEMM** 是所有問題 shape 都相同的特例。
- **G**：一次呼叫裡的問題個數。
- **N**：一個問題用的模數個數，範圍 [2, 20]（`GGEMM_MAX_MODULI`）。
- **p_1…p_N**：兩兩互質、每個都 ≤ 256 的模數。**P** = ∏p_l。
- **CRT**：中國剩餘定理（Chinese Remainder Theorem）。
- **Kp** = round_up(k, 32)。
- **plane**：一個模數對應的一份 INT8 residue 矩陣。N 個模數的問題實際有 N+1 份 plane，多出的一份是 abs plane（§2.5）。
- **u** = 2^-53。**γ_n** = n·u/(1−n·u)。
- **φ（phi）**：測資動態範圍參數，a_ij = (U(0,1)−0.5)·exp(φ·N(0,1))，沿用 Ozaki 論文的慣例。φ 越大，同一列或同一行內的指數範圍越寬。
- **H2D**：host 到 device 的複製。**SM**：Streaming Multiprocessor。**CTA**：thread block。
- **LPT**（Longest Processing Time first）：工作量由大到小排序的排程法。
- **MoE**：Mixture of Experts。**LoRA**：Low-Rank Adaptation。**BSR**：Block Sparse Row。
- **TOPS**／**TFLOPS**：每秒 10^12 次整數／浮點運算。
- **NVTX**：NVIDIA Tools Extension 的標記區段。**nsys**：Nsight Systems。**ncu**：Nsight Compute。

v1 的範圍：只支援 column-major、不轉置，只支援 FP64 實數。

---

## 2. 方法

### 2.1 Ozaki scheme II（CRT，INT8）

依 Ozaki、Uchino、Imamura 的論文（arXiv:2504.08009）自行撰寫，程式在 `src/ozaki2_numerics.cu`。流程分五步：

1. **Scaling**：A 每一列取 2^{s_i}，B 每一行取 2^{t_j}。s_i 取能滿足 2^{s_i}‖A(i,:)‖₂ ≤ T 的最大值，其中 T = sqrt(P/2·(1−2^-20))·(1−2^-40)（論文的 fast mode，用 Cauchy–Schwarz 界）。然後令 A' = trunc(2^{s_i}A)、B' = trunc(2^{t_j}B)，一律向零截斷。這保證 Σ|A'||B'| < P/2，CRT 還原不會有歧義。
2. **Residue**：把 mantissa 拆成三段 18-bit，配合查表的 [2^e]_p，再用 Barrett 約簡求餘數，最後存成對稱範圍的 int8。
3. **INT8 GEMM**：每個模數做一次 C_l = A'_l·B'_l，以 INT32 精確累加。要求 Kp·128² < 2^31，所以 k ≤ `OZ2_MAX_K` = 131040。
4. **CRT reconstruct**：把 S = ΣM_l·y_l 拆成 uint64 + uint32 兩部分精確累加，Q = rint(S/P)，相當於 double-double 等級的運算。
5. **反 scale**：C = α·2^{−(s+t)}·X + β·C。乘 2 的冪次沒有捨入誤差。

N 的選擇（job 590258 `logs/test_numerics_standalone.log`）：normwise 目標 1e-15 時，k = 256／1024／4096 取 N = 15，k = 16384／65536 取 N = 16；k = 131072 時回傳 −1，代表做不到，必須 fallback。

### 2.2 Grouped pipeline：整組固定 3 個 kernel

`EMU_GROUPED`（`src/ggemm.cu`）的流程：

```
host：planner + descriptor 打包（pinned buffer）→ 一次 descriptor H2D
GPU ：oz2_prep_kernel → i8_grouped_persistent_kernel → oz2_reconstruct_kernel（各 1 次 launch）
```

不論 G 多大，launch 數都固定是 3。job 511249 印出 `emu_grouped launches: G=16 -> 3, G=1024 -> 3`。例外：只要組內有 k = 0、α = 0 的問題（或 AUTO 的 native fallback），就會多一次 cuBLAS 呼叫（例如 `zero_dims_mix` 的 launches = 4）。split-K cfg3 無法經由 ggemm API 啟用（§2.3），所以不是第 4 個 launch 的來源。所有 workspace 都在 `ggemm_create` 時配好；計時區內不做 allocation，workspace 不夠就回傳 `GGEMM_ERR_WORKSPACE`。

### 2.3 Persistent grouped INT8 GEMM

程式在 `src/int8_grouped_gemm.cu`，沒有用 CUTLASS：

- 使用 `mma.sync.m16n8k32 s8.s8.s32`、3-stage `cp.async`、`ldmatrix`，shared memory 以 XOR-swizzle 避免 bank conflict。
- 三種 tile config 在同一個 kernel 內：cfg0 128×128（BK 64）、cfg1 64×64（BK 128）、cfg2 32×32（BK 256）。三者共用 48 KB smem，每 thread ≤ 128 registers，所以每個 SM 能放 2 個 CTA。
- **LPT 排程**：host 依每個 tile 的成本由重到輕排序，並建立 tile prefix sum。persistent CTA 從 atomic counter 取 tile id（提前預取一個），再用二分搜尋找出 tile 屬於哪一題。最後一個 CTA 把 counter 歸零，所以不需要額外的 memset launch。
- **tile config 只有 AUTO 會逐題選**；EMU_GROUPED 與 EMU_GROUPED_REUSE 固定用 cfg0（128×128）。LPT 排程則是所有 grouped path 都有。
- split-K 32×32（cfg3）是獨立的 kernel。auto split-K 規則（`i8_set_auto_splitk_min_kp`）只對 `tile_cfg < 0` 的問題生效（`src/int8_grouped_gemm.cu` 第 614–619 行），而 ggemm 的 planner 一律指定 tile_cfg 0..2（`src/ggemm.cu` 的 `d.tile_cfg = e.tile_cfg`），**所以經由 ggemm API 開不了 split-K**，只有直接呼叫 `i8_grouped_gemm` 的測試會用到。效能評估見 §9.3。

### 2.4 Prep 與 reconstruct

- **prep**：一次 launch 處理整組。它算 2-norm 與 scale，寫出 N+1 份 residue plane 和 row/col workspace（內容包括 s_i、σ_i、‖dA_i‖ 上界、‖A'_i‖ 上界）。最終 commit b56631d 加入 **prep split-K**，即沿 k 切段計算再合併。job 590258 驗證 S = 2、3、7、16、64 和 S = 1 的結果相比是 `0 differing buffers`（bit-exact）。
- **reconstruct**：一次 launch 處理整組，做 CRT 還原、反 scale、套 α/β，以及 §2.5 的認證。

### 2.5 Certify-and-repair

normwise 誤差界無法保證 componentwise 誤差：ρ_ij = ‖a_i‖‖b_j‖/(|A||B|)_ij 可以任意大，所以沒有任何固定的 N 能讓每個 entry 都達到 FP64 的 componentwise 界。我們的做法是逐 entry 認證：

1. 在 reconstruct 中算出每個 entry 的嚴格誤差上界 E_ij。
2. 用多出來的 **abs plane** 算 (|A||B|)_ij 的嚴格下界：A'' = floor(2^{σ}|A|) ∈ [0,127]，以 INT32 精確算出 L_ij = (A''B'')_ij。
3. 若 |α|·E_ij·(1+2u) ≤ (γ_n−2u)·(|α|·Lab_ij + |β·C0_ij|)，該 entry 就通過認證。沒通過的 entry 依 repair mode 處理：
   - **OFF**：不認證。
   - **COUNT**：保留模擬值，只計入 `num_uncertified`。這是 `EMU_*` 路徑的預設，所以計時區內是純模擬。
   - **ON**：在同一個 thread 內改用 FP64 FMA 內積重算，計入 `num_repaired`。這是 `AUTO` 的預設。
4. 不論哪個 mode，含 NaN/Inf 的列或行一律走 FP64 dot（safety fallback）。

**計數器的實作**（commit 1c39e40）：uncertified、repaired、safety_fallback 三個計數器原本是「每個計數非零的 thread 各對同一個全域位址做一次 atomicAdd」。現在改成先在 warp 內聚合（warp shuffle），再在 block 內聚合，最後每個 block 只做一次全域 atomicAdd；prep 的 NONFINITE 計數改用 ballot。數值結果與計數完全不變，效果見 §9.6。

例子（job 590258 numerics log，64 個 16³ 問題，φ = 2，N = 15）：

| mode | gate | 計數 |
|---|---|---|
| OFF | 5.919 | — |
| COUNT | 5.919 | uncertified = 899 |
| ON | 0.2866 | repaired = 899 |

OFF 和 COUNT 的輸出 bitwise 相同。gate 的定義見 §5.1，≤ 1 才算達標。

### 2.6 Planner／AUTO

程式在 `src/planner.cpp`。每個問題依下表決定去向：

| route | 條件 | 去向 |
|---|---|---|
| EMPTY／K_ZERO | m·n = 0 或 k = 0 | 不算，或只算 β·C |
| ALPHA_ZERO | α = 0 | cuBLAS 算 β·C |
| K_TOO_LARGE | k > 131040 | native；若 `allow_fallback = 0` 回傳 `GGEMM_ERR_UNSAFE` |
| SMALL_K | AUTO 且 k ≤ 16 | native |
| TINY | m·n·k < 32³（`GGEMM_TINY_MNK`） | native |
| UNPROFITABLE | cost model 預測 t_emu > t_native | native |
| EMU | 其餘 | 模擬，每題各自決定 N 與 tile config |

- **N**：AUTO（adaptive N）用 `oz2_choose_num_moduli_cw(k, ρ_budget)`，ρ_budget 預設 16，**這個值沒有調過**。`GGEMM_RHO_BUDGET` 是 function-static，整個 process 第一次 plan 時讀一次，而且只用在這裡。`EMU_*` 路徑則用 `target_rel_err`（預設 1e-15）。
- **Tile**：在 padding 效率 ≥ 0.75 的前提下取最大的 tile，再做 wave-quantization 調整。
- **Cost model**：t_emu = prep 流量/BW + ⌈planes·tiles/slots⌉·per_tile + reconstruct 流量/BW；t_native = max(2mnk/FP64 峰值, 流量/BW)。參數取自 `bench/calib/params_nvidia_h200.json`。
- 每次呼叫走 native 的題數會寫在 `stats.num_fallback`。
- **Per-call shape memo**（commit a48eb3e）：在同一次呼叫內，α = 0 檢查之後的規劃結果只取決於 (m, n, k, β ≠ 0)，所以相同 shape 的問題只規劃一次（`ggemm_plan_ex` 搭配 memo 表，大小由 `ggemm_plan_memo_cap(G)` 決定），呼叫路徑中沒有配置。`tests/test_planner.cpp` 第 403 行起用隨機 group 比較 memo 與非 memo 的 plan，要求 bit for bit 相同。
- 效能與測試（job 590223，`planner_unit_590223/test_planner.log`，commit a48eb3e）：
  - planner 單元測試 **1655 passed / 0 failed**，包含 memo 與非 memo 的逐 entry 比對。
  - G = 65535 時，非 memo 的 `ggemm_plan` host 時間平均為 path 3 7322.2 µs（111.02 ns/problem）、path 4 8214.4 µs（124.81 ns/problem）。
  - `operator new` 呼叫次數：首次呼叫與 steady state 都是 0。
  - 舊版（job 511063）是 1607/1607，path 3 為 110.17 ns/problem。

### 2.7 Operand reuse（path 5）

程式在 `src/operand_reuse.cpp`。找出組內共用同一個 A 或同一個 B 的問題（同一個指標、同一個 shape），每個不同的 operand 只做一次 prep。`EXACT_N` 政策只在 N 相同的問題之間共用，所以結果和 `EMU_GROUPED` bit-exact。設定 `GGEMM_REUSE_PROMOTE=1` 會把可共用的問題提升到同一個 N，換取更多共用。

### 2.8 CUDA Graph plan/execute

程式在 `src/ggemm_graph.cu`。`ggemm_make_plan` 在計時區外做完 planner、descriptor H2D、capture、instantiate 與 `cudaGraphUpload`。`ggemm_execute_plan` 只做一次 `cudaGraphLaunch`。`ggemm_plan_update_pointers` 直接 patch device descriptor，不需重新 instantiate。只有 path 3、4 可以建 plan；plan 內不做 reuse；shape、α、β 都固定在 plan 裡。

### 2.9 七條路徑與各自的設定（同一個 API 切換）

bench／test 對各路徑使用的設定不同（`tests/test_grouped.cu: cfg_for`、`src/planner.cpp`）。後面所有表格都依這張表解讀。

| # | 路徑 | 說明 | 選 N 的方式 | repair mode | allow_fallback | 正確性 gate（§5.1） | 角色 |
|---|---|---|---|---|---|---|---|
| 0 | NATIVE_LOOP | 每題一次 `cublasDgemm` | — | — | — | strict | native baseline |
| 1 | NATIVE_GROUPED | `cublasDgemmGroupedBatched` | — | — | — | strict | 主要的 native 比較對象 |
| 2 | EMU_LOOP | 逐題模擬，每個 plane 各一次 INT8 launch，共 O(G·N) 次 | normwise（target 1e-15） | COUNT | 0 | cert | 自寫的 naive starter-like baseline |
| 6 | EMU_LOOP_BATCHED | 每題 3 次 launch | normwise | COUNT | 0 | cert | GEMMul8 風格逐題 baseline（**自己寫的**，不是 GEMMul8 程式碼） |
| 3 | EMU_GROUPED | 整組 3 個 kernel | normwise | COUNT | 0 | cert | 最終版：純模擬，量 grouping 的增益 |
| 5 | EMU_GROUPED_REUSE | 3 + operand reuse | normwise | COUNT | 0 | cert | 量 reuse 的增益 |
| 4 | AUTO | 3 + planner | componentwise（`oz2_choose_num_moduli_cw`，ρ_budget 16） | **ON** | **1**（唯一允許 fallback 的路徑） | strict | 預設路徑；在 H200 上實質等於 native（§10.1） |

**關於 starter**：沒有官方 starter code。`src/ggemm.cu` 第 614–615 行的註解寫明「Starter-like baseline (no starter source available to us, so we model the usual single-GEMM emulation)」。規格頁從叢集上連不到，GitHub 上也找不到官方 starter。因此本報告用兩種方式代表「starter」：
- **starter = v0**：我們第一個可用的版本（commit 5d8e2ad）。v0 的公開 header 已經把 `GGEMM_PATH_EMU_LOOP` 標為「starter-like: single-GEMM INT8 emulation called per problem」（`git show 5d8e2ad:include/ggemm.h` 第 23 行）。v0 與最終版在同一個 job 內的比較見 §7.0。
- **最終版內的 starter-like 路徑**：自寫的 `EMU_LOOP`，以及更強的 `EMU_LOOP_BATCHED`。它們和最終版共用同一組 numerics kernel，只差在 launch 怎麼組織（§7.1）。

---

## 3. 實驗環境

| 項目 | 值 | 來源 |
|---|---|---|
| 叢集 | NCHC nano4，partition `dev`，每 job 1 GPU、12 CPU | `scripts/final/run_final.slurm` |
| GPU | NVIDIA H200（sm_90，132 SMs，143771 MiB） | job 590258 `provenance.txt`；校正 JSON |
| Driver | 580.65.06（CUDA Version 13.0） | job 590258 `nvidia-smi.txt` |
| CUDA | 13.0，nvcc V13.0.48（`cuda_13.0.r13.0/compiler.36260728_0`） | job 590258 `nvcc.txt` |
| Clocks | SM／Graphics 最高 1980 MHz，Memory 3201 MHz；Applications Clocks 為預設值 1980 MHz，**未鎖頻**；power limit 700 W | job 590258 `nvidia-smi.txt` |
| compute-sanitizer | 2025.3.0.0 | job 511062 |
| 編譯 | sm_90 與 sm_120 都編，`-O3 -lineinfo`，C++17 | `scripts/build.sh` |
| 正式版本 | commit 1c39e40（dirty=6，只有文件與未追蹤項目，`src/` 乾淨） | job 590258 `provenance.txt` |
| 節點 | 590258：25a-hgpn069；590147：25a-hgpn004；590148：25a-hgpn042 | 各 `provenance.txt` |

**目標平台說明**：題目的目標機是 RTX 5090（sm_120，FP64 很弱），**但 nano4 上沒有 RTX 5090**。所有 job 都在 H200 上執行。程式雖然有編 sm_120，卻**從未在 sm_120 上執行過**，所以本報告不對 5090 的效能做任何宣稱。

**校正參數**（`bench/calib/params_nvidia_h200.json`，擬合自 `calib_510850.csv`）：

| 參數 | 值 |
|---|---|
| t_launch | 2.0676e-06 s |
| DRAM BW | 3.944981e12 B/s |
| FP64 擬合峰值（cuBLAS DGEMM） | 6.1326e13 flop/s |
| INT8 擬合峰值（cuBLASLt） | 1.67606e15 op/s |
| INT8 實測最大值（`i8_peak_ops_max`） | 1.50358e15 op/s |

INT8 峰值是用 cuBLASLt 擬合的，我們自己的 kernel 只有它的約 40%（§8.3），所以 cost model 對模擬的估計偏樂觀。

---

## 4. 測資與問題規模的選擇理由

Grouped 路徑的價值有兩點：把每次呼叫的固定成本攤到很多題上，以及在單題太小或大小不均時仍讓 INT8 Tensor Core 保持忙碌。每個家族各對應一種逐題迴圈會吃虧的實務情境（完整理由見 `bench/workloads.cu` 檔頭）。

| 家族 | 參數（`bench` set） | 理由 |
|---|---|---|
| uniform small | G ∈ {256, 1024, 4096} × s ∈ {16, 32, 64, 128, 256}，去掉 G = 4096、s = 256（超過單一 operand 2^26 個元素的上限） | 經典 batched BLAS（FEM、block-sparse、tensor network）。每題 2s³ 從 8K flops 到 33M flops，可以看出模擬從哪裡開始划算或不划算 |
| MoE | E = 8 top2（1024／3584）、E = 64 top6（2048／1408）、E = 128 top8（2048／768）；prefill 與 decode（T = 64）；Zipf s ∈ {0, 1.2}；up 與 down 兩種 projection | m_e 偏斜，也可能是 0；expert 順序用 seeded permutation 打亂，避免 in-order 排程佔便宜 |
| hetero | G ∈ {64, 256}，m、n、k 各自 log-uniform 取自 [16, 2048] | 最大題與最小題差約 10^6 倍，測 tile 選擇與 LPT |
| LoRA | G = 16 r = 16、G = 64 混合 rank {8, 16, 32, 64}；T = 2048，hidden 4096；shrink（n 小）、expand（k 小） | Punica／S-LoRA 的情境；n 小時方形 tile 失效，k 小時 reconstruct 成為主要成本 |
| 應用 MoE FFN | T = 4096、E = 16、top2、1024／2048、Zipf {0, 1.0, 1.5}；另加 T = 16384、E = 8、top1、256／256、Zipf 1.2 | 科學計算 ML 的 FP64 MoE，例如 Behler-Parrinello／DeePMD 類型原子勢（等同 top-1 MoE），**不是** LLM；router 是合成的 Zipf |
| 應用 block-sparse | (atoms, basis, nrhs) = (1024, cc-pVDZ, 128)、(512, cc-pVTZ, 256)、(256, cc-pVQZ, 256)，cutoff 6 Å，block 與 panel 兩種模式 | 線性尺度電子結構（DBCSR／CP2K 類）的 F·C_occ，大量大小不一的小 FP64 GEMM |

job 590258 的 `bench` set 共 44 個 workload（`shapes.csv`：uniform 14、MoE 24、hetero 2、LoRA 4），φ = 0.5。正確性測試用縮小版的 `test` set，另外加 k = 4096、k 不是 32 倍數兩個案例，並掃 φ ∈ {0, 0.5, 1, 2}。

---

## 5. 正確性標準與結果

### 5.1 標準

**Reference**：在 GPU 上用 double-double 計算 Cref = α·A·B + β·C0（`tests/reference.h`），不拿 native FP64 當標準答案。

**指標**：
- den_ij = |α|(|A||B|)_ij + |β||C0_ij|。
- **err_cw** = max |C−Cref| / den。
- n_g = k_g，α ≠ 1 時 +1，β ≠ 0 時再 +1。
- **err/gamma** = max_g err_cw_g / γ_{n_g}。≤ 1 表示達到 native DGEMM 自己滿足的界（Higham 2002 §3.5）。
- **over_gamma**：超出界的 entry 數，non-finite 也算。
- **vs_native** = err_cw / cuBLAS 在同一份資料上的 err_cw。
- **rel_max** 只列出供參考，不參與判定。

**正確性有兩種 gate**：
- **strict**：適用 native 路徑與 AUTO（repair ON）。要求每個 entry 都滿足 err_ij ≤ γ_n·den_ij，也就是 err/gamma ≤ 1，而且沒有 non-finite。
- **cert**：適用 `EMU_*` 路徑（repair COUNT）。要求 n_over_gamma ≤ num_uncertified，也就是**每個超過 γ_n 的 entry 都必須被 certificate 標成 uncertified**。certificate 可以多報，**不能漏報**。**cert gate 並不要求 EMU 的輸出達到 FP64 等級**：EMU 路徑可以有 err/gamma > 1 的 entry，只要都被標記出來即可（實際數字見 §5.3）。

**Git 歷史中的變更**（`git log -p -- tests/reference.h tests/reference.cu tests/test_grouped.cu`）：
- γ_n 的定義（`fp64_gamma`、`fp64_bound_terms`）和 double-double reference 從 5d8e2ad 至今都沒有改過，也就是**門檻與 reference 從未放寬**。
- 但 EMU 路徑適用的 gate 改過：5d8e2ad 時所有路徑都用 strict（`ok = nonfinite == 0 && ratio_gamma <= 1.0`）。173bb94 引入 repair COUNT 後，EMU 路徑改成 cert。job 510881 的 26 個 failure 中，有一部分就是 EMU 在 φ = 2 時沒過 strict gate（§10.6）。
- 唯一可調的 tolerance 是 `target_rel_err`（預設 1e-15），它只影響 `EMU_*` 選的 N。

**安全性**：計時區內不 allocate；workspace 不夠時回傳錯誤碼；fallback 數看得到；另外跑 compute-sanitizer（§10.3）。

### 5.2 結果

| 測試 | 結果 | 來源 |
|---|---|---|
| test_grouped（7 條路徑 × test set × 4 種 φ × α/β 組合 + edge + big-K） | **2955 passed / 0 failed** | **job 590256 與 590258（commit 1c39e40）**；較早的 511249（b56631d）、590147、590162 也都是 2955/0 |
| test_grouped 在 memcheck 下（縮小版） | 312 passed / 0 failed，`ERROR SUMMARY: 0 errors`，leak 0 bytes | job 511249 `test_511249/memcheck.log`；四種 sanitizer 工具的全面重跑見 §10.3（job 590161、590498） |
| test_graph（plan 和 direct bit-exact、換 stream、update_pointers） | **63 / 0** | job 590258（早先的 510998、511060、590147 也是 63/63） |
| test_reuse_e2e（REUSE 和 GROUPED bit-exact） | **54 / 0** | job 590258；舊版（590147）是 30/0；job 511060 另含 PROMOTE |
| test_planner（含 memo 與非 memo 的逐 entry 比對） | **1655 / 0** | job 590223（a48eb3e）；舊版 511063 是 1607/0 |
| test_numerics_standalone | **PASSED (0 failures)** | job 590258；job 511101 |
| test_i8_grouped（對 CPU、cuBLASLt bit-exact；kp = 131040 邊界） | **ALL PASS** | job 590258 |

Edge cases（job 511249）：1×1×1、zero_dims_mix、all_empty、beta0_nanC、k_not_mult32 全部 ok。big-K（k = 140000）：AUTO 有 3 題 fallback，err/gamma = 0.1885；`allow_fallback = 0` 時 emu_grouped 與 auto 都回傳 rc = 3（UNSAFE），符合預期。non-finite 測試：counter = 2，預期也是 2（job 590258）。

**逐路徑彙總（test set，所有 φ 與 edge cases；job 590258 `tables.md` §1，來源 `tests.csv`）：**

| path | gate | cases | pass | fail | worst err/gamma | worst vs native | nonfinite | fallback (sum) |
|---|---|---|---|---|---|---|---|---|
| native_loop | strict | 66 | 66 | 0 | 0.5584 | 1.000 | 0 | 0 |
| native_grouped | strict | 66 | 66 | 0 | 0.5584 | 8.502 | 0 | 0 |
| emu_loop | cert | 66 | 66 | 0 | **175.9** | 445.5 | 0 | 0 |
| emu_loop_batched | cert | 66 | 66 | 0 | **175.9** | 445.5 | 0 | 0 |
| emu_grouped | cert | 66 | 66 | 0 | **175.9** | 445.5 | 0 | 0 |
| emu_grouped_reuse | cert | 66 | 66 | 0 | **175.9** | 445.5 | 0 | 0 |
| auto | strict | 66 | 66 | 0 | 0.5584 | 8.502 | 0 | 3778 |

**Full bench set 的精度**（不計時的 check call，φ = 0.5，44 列／路徑；`tables.md` §1，來源 `bench.csv`）：native 與 auto 的 err/gamma > 1 都是 0 列，最大 0.5055。四條 EMU 路徑各有 1 列 err/gamma > 1，最大 2.204，worst rel_max 6.273e-09。590258 的 bench 在一次不計時的 `--count-call` 中讀了 certificate 計數（native 路徑沒有 certificate，所以是 −1，屬於正常）。EMU 路徑中有 28 列 num_uncertified > 0，**沒有任何一列 n_over_gamma > num_uncertified**（由 `bench.csv` 統計），所以上面那一列超出 γ 的 entry 也都被標記出來了。

**Certificate repairs**（`tables.md` §1）：
- `test_reuse_e2e`（`GGEMM_STATS_REPAIR=1`，3 個 workload）：所有路徑 repaired 都是 0。
- `test_numerics_standalone`：per-shape × N sweep 共 260 列，260 ok／0 FAIL，worst gate 0.3035，repaired 共 643030 個 entry；grouped launch 有 6688 個 entry 被重算。

### 5.3 純模擬（COUNT）不一定達到 FP64 界，以及 repair 的頻率與成本

`EMU_*` 路徑通過的是 cert gate，不是 strict gate。

**EMU 路徑 err/gamma > 1 的列數**（job 590258 `tests.csv`；四條 EMU 路徑合計 264 列；由 tests.csv 統計）：

| φ | 列數 | err/gamma > 1 的列數 | 最大 err/gamma |
|---|---|---|---|
| 0 | 48 | 0 | 0.2363 |
| 0.5 | 48 | 4 | 1.001 |
| 1 | 96 | 24 | 4.686 |
| 2 | 48 | 24 | **175.9** |
| edge cases | 24 | 4 | 4.127 |
| 合計 | 264 | 56 | 175.9 |

這 56 列都通過 cert gate，也就是超出 γ 的 entry 全部被標成 uncertified。但它們的輸出**不是** FP64 等級。job 511249 的個別例子：

| case | 路徑 | err/gamma | over_gamma | uncertified |
|---|---|---|---|---|
| lora_G8_rmix_expand φ = 1 | emu_grouped（COUNT） | 4.686 | 138 | 22655 |
| lora_G8_rmix_expand φ = 2 | emu_grouped（COUNT） | 175.9 | 13561 | 113944 |
| 同上 | auto（ON + fallback） | 0.5232 | 0 | 0 |

要真正達到 FP64 等級，必須用 repair ON。certificate 是保守的：φ = 2 時多報了 113944/13561 ≈ 8.4 倍。

**Repair sweep**（正式版 job 590257，commit 1c39e40，`results/repair_590257/plots/repair_summary.md`；EMU_GROUPED、固定 N、COUNT 與 ON 比較；計時 R = 10、warmup 3 取 median；uncert frac 來自另一次不計時的呼叫）。下表取 N = 15，括號內是 COUNT 下的 uncertified 比例：

| workload | φ=0 ON/COUNT | φ=0.5 | φ=1 | φ=2 | φ=2 時 err/gamma（COUNT → ON） |
|---|---|---|---|---|---|
| uni_G1024_s64 | 0.995 (0) | 1.004 (0) | 0.990 (9.5e-07) | **1.186** (4.60e-02) | 10.87 → 0.2295 |
| uni_G256_s256 | 0.999 (0) | 1.000 (0) | 0.997 (0) | **2.902** (3.51e-02) | 2.784 → 0.1128 |
| moe_E64_prefill_z1.2_up | 1.000 (0) | 0.999 (0) | 1.001 (0) | **4.113** (2.04e-02) | 0.7805 → 0.02758 |
| het_G256_16-2048 | 1.000 (0) | 1.000 (0) | 0.999 (2.2e-06) | **3.337** (3.76e-02) | 98.36 → 0.4012 |

舊版 job 590148（b56631d）和 590257 的 192 列 EMU 結果，以 (workload, path, G, φ, N, repair_mode, ρ, total_entries) 配對後，所有計數與 err_max 都完全相同（§9.6）。因此下面的重算比例兩個 job 一樣。

**repair ON 時的 FP64 重算比例**（job 590257 與 590148 的 `emu_repair.csv`，repair_mode = on 共 96 列；比例 = num_repaired / total_entries，由 CSV 統計）：

| 分組 | 各列比例的平均 | 最大 |
|---|---|---|
| 全部 96 列 | 13.2%（若以 entry 總數加權則是 13.3%） | 100% |
| φ = 0 | 5.9% | 99.98% |
| φ = 0.5 | 7.6% | 100% |
| φ = 1 | 12.9% | 100% |
| φ = 2 | 26.3% | 99.99% |
| N = 13 | 64.5% | 100% |
| N = 14 | 13.6% | 57.0% |
| N = 15 | 0.87% | 4.60% |
| N = 16 | 0.03% | 0.16% |
| N = 18 | 0.00% | 0.028% |
| N = 20 | 0.01% | 0.037% |

N = 13 時，uni_G1024_s64 在 φ = 0／0.5／1／2 的重算比例分別是 99.983%、99.996%、100.000%、99.992%，**答案實質上是 FP64 算出來的**。所以 **repair ON 是安全保證，不是效能手段**。本報告所有 EMU 的效能數字（表 2a–2c、§7、§9.1）都是 COUNT 模式量的。

整張 sweep 表的結論：
- ON 模式的 err/gamma 在所有格子都 ≤ 1，最大是 0.4559（het，φ = 2，N = 13）。COUNT 模式最差到 1.018e+04（同一格）。
- 最貴的情況是 ON/COUNT = 5.634（moe，φ = 2，N = 13；590257）。COUNT 在計數器修正後變快了，所以 repair 的相對成本看起來比舊版（590148）的 4.684 更高。
- **φ = 2 時，加大 N 比修復更划算**（590257）：het 的 ON 時間在 N = 15 是 34.71 ms，N = 18 只要 12.60 ms；moe up 在 N = 15 是 28.03 ms，N = 18 是 8.630 ms。
- 另一份資料（job 511101 `numerics_perf_511101.log`，ON，N = 15，被重算的 entry 數）：φ ≤ 1 的四種 shape 都是 0；φ = 4 時 128×96×1000 有 9331/12288 ≈ 76% 被重算，64×64×4096 有 3306/4096 ≈ 81%。

這些數字支持 README 的說法：動態範圍寬時付出的是速度，不是正確性。

---

## 6. 量測方法（native、starter、最終版本在相同條件下比較）

### 6.1 共同條件

依 `bench/bench_grouped.cu` 檔頭：

- 同一 workload、同一份資料（固定 seed）、同樣的 handle sizing、同一個 stream；不同路徑只換 path 參數。
- 所有 allocation 都在計時區外。**實測證據**：bench 在每次計時呼叫前後計算 host 配置次數，寫在 `host_allocs_timed` 欄。job 590258 與 590224 的 `bench.csv` 各 308 列（44 個 workload × 7 條路徑），全部是 0。
- **warmup 5 次、計時 20 次，取 median**（另外也輸出 min 和 wall）。job 590258 用 `REPS=20 WARMUP=5`。
- β = 0，所以重複呼叫是 idempotent 的。不 flush L2。
- `effective_GFLOPS` 只計入被取代的 FP64 flops。
- 誤差和 certificate 計數來自另一次不計時的呼叫；計數用的是另一個設了 `GGEMM_STATS_REPAIR=1` 的 handle。

### 6.2 時間種類（全文固定使用以下名稱）

| 名稱 | 定義 | 包含什麼 | 來源 |
|---|---|---|---|
| **E2E** | `bench.csv` 的 `time_ms`：用 cudaEvent 量單次 `ggemm_grouped_dgemm()`，取 20 次的 median | host planning、descriptor 打包、descriptor H2D、所有 kernel | `bench_grouped`（§7.1）；應用層的 E2E 是 `total_ms_median`，包含 route／gather 等階段（§7.3） |
| **GPU-busy** | nsys 量到每次呼叫的 GPU kernel 與 memop 時間（`busy_ns`）及 span（`span_ns`），5 次 timed call 的 median | 只有 GPU 時間 | `analysis/nsys_summary_*.csv`（§8）。ncu 的 kernel duration 是在 replay 下量的，只看相對大小，**不**和 E2E 比較 |
| **Graph replay** | `test_graph` 的 `gpu_us`：cudaEvent 量 R = 200 次 `ggemm_execute_plan` 的平均；另外附 host_issue_us、make_plan_ms | graph 內的 kernel | §7.5 |
| **reuse** | `bench.csv` 中 `emu_grouped_reuse` 那一列的 E2E | 同 E2E | §7.4。**不引用** `test_reuse_e2e` 印出的時間，因為它的計時區內含同步，正在修正 |
| （附）INT8 kernel | `test_i8_grouped`：3 次 warmup + 20 次計時取平均；TOPS = 2·m·n·kp/t | 只有 INT8 kernel | §8.3、§9 |

Speedup 一律定義為 time(base)/time(path)，> 1 表示 path 比較快。凡是有 AUTO 的表格都會一併標出它的 fallback 比例（fallback 題數 / G）。**AUTO 相對任何 EMU 路徑的 speedup 都不能算成模擬的加速**，因為在 H200 上 AUTO 實際執行的是 cuBLAS。

**量測順序效應（§10.8）**：在 MoE decode（E64、E128）上，native_grouped 的時間取決於它在同一個 bench process 裡是否第一個被量。bench 矩陣中排在其他路徑後面的 native_grouped 會被高估。本報告對這些列的 native baseline，改用「第一個量測位置」的數字（job 590956）。其他 workload 不受影響。

**starter／最終版比較的條件**（job 590633，`scripts/final/starter_vs_final.slurm`，`results/v0final_590633/compare.md`）：
- v0（5d8e2ad）與最終版在同一個 job 內編譯，並在同一張 GPU（25a-hgpn038）上執行。
- 兩版用同樣 44 個 workload，reps 20、warmup 5。
- v0 與最終版交替各跑 2 次，取 2 次的 median。
- 最終版 binary 的 commit 是 e20a669；它的 `src/`、`include/` 與 1c39e40 相同（`git diff 1c39e40 e20a669 -- src include` 為空）。

---

## 7. 效能結果

### 7.0 native、starter（v0）、最終版的三方比較（job 590633）

v0/final 的 geomean（44 列，> 1 表示最終版比較快；`compare.md` 的「Geomean speedup per path」）：

| path | geomean v0/final | min | max | 說明 |
|---|---|---|---|---|
| auto | 1.20× | 0.96× | 2.37× | v0 的 AUTO 同樣全部 fallback，加速來自 planner shape memo（§9.6） |
| emu_loop（starter-like） | 1.14× | 0.87× | 1.63× | 主要來自 prep split-K |
| emu_grouped | **0.91×** | 0.76× | 1.35× | **最終版慢 9%**，見下面說明 |
| native_grouped | 0.95× | 0.62× | 1.09× | 不是 regression：MoE decode 列（0.62×–0.82×）是量測順序假象，§10.8 |
| native_loop | 0.99× | 0.94× | 1.02× | 兩版都直接呼叫 cuBLAS |

**同一個問題上的 starter 與最終版**（`compare.md` 最後一張表；比值由表中數字計算）：
- v0 EMU_LOOP（starter-like）÷ 最終版 EMU_GROUPED 的 geomean 是 **13.86×**，範圍 2.04×–91.2×。各家族 geomean：uniform 58.9×、MoE 6.78×、hetero 6.32×、LoRA 9.49×。
- 最終版 native_grouped ÷ 最終版 EMU_GROUPED 的 geomean 是 0.135×。也就是說，不論 starter 或最終版，模擬在 H200 上都比 native 慢。

**為什麼最終版 EMU_GROUPED 比 v0 慢 9%**：
- 兩版選的 N 相同。44 列的 `err_max` 逐位元相同（`compare.md` 的 v0 err_max 欄與 final err_max 欄）。
- 最終版每題多算一份 abs plane，也就是 N+1 份 INT8 GEMM；N = 15 時 INT8 工作量多 1/15 ≈ 6.7%。
- 最終版還做逐 entry 的誤差認證（§2.5）。v0 沒有任何誤差認證，所以**無法偵測不準的輸出**；例如 φ = 2 時純模擬的 err/gamma 可以到 175.9（§5.3），v0 不會回報。
- 所以這 9% 是**安全保證的代價**。abs plane 與認證各占多少，**沒有分開量測**（上面的 6.7% 只是工作量推論）。
- 例外：lora shrink 與 moe_E8_prefill_down 上最終版較快（1.05×–1.35×），與 prep split-K 有關（推論，未分開量測）。

### 7.1 Full bench set 的 E2E（正式 job 590258，φ = 0.5，adaptive N，ms）

設定（§2.9）：EMU 路徑用 normwise 選 N、repair COUNT、allow_fallback = 0；AUTO 用 componentwise 選 N、repair ON、allow_fallback = 1。

表中欄位（數字取自 `tables.md` 表 2a、2b、2c 與 `bench.csv`）：
- **ELB/EG** = time(emu_loop_batched) / time(emu_grouped)，即純模擬之間的 speedup。
- **EG 相對 NG** = time(native_grouped) / time(emu_grouped)，< 1 表示模擬比較慢。
- **auto 相對 NG** = time(native_grouped) / time(auto)。
- **auto fallback** = 走 native 的題數 / G。

| workload | G | native_grouped | emu_loop | emu_loop_batched | emu_grouped | emu_grouped_reuse | auto | auto fallback | ELB/EG | EG 相對 NG | auto 相對 NG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uni_G256_s16 | 256 | 0.02341 | 23.8 | 6.478 | 0.2476 | 0.2782 | 0.02506 | 256/256 | 26.16× | 0.09× | 0.93× |
| uni_G256_s32 | 256 | 0.02459 | 28.05 | 8.92 | 0.2967 | 0.3378 | 0.02656 | 256/256 | 30.07× | 0.08× | 0.93× |
| uni_G256_s64 | 256 | 0.02757 | 30.73 | 9.796 | 0.4244 | 0.4774 | 0.02942 | 256/256 | 23.08× | 0.06× | 0.94× |
| uni_G256_s128 | 256 | 0.06816 | 36.33 | 11.37 | 0.8083 | 0.889 | 0.06984 | 256/256 | 14.07× | 0.08× | 0.98× |
| uni_G256_s256 | 256 | 0.3505 | 44.08 | 14.47 | 2.533 | 2.66 | 0.3522 | 256/256 | 5.71× | 0.14× | 0.99× |
| uni_G1024_s16 | 1024 | 0.06392 | 97.72 | 26.16 | 0.9725 | 1.083 | 0.07013 | 1024/1024 | 26.90× | 0.07× | 0.91× |
| uni_G1024_s32 | 1024 | 0.06762 | 115.6 | 36.07 | 1.166 | 1.321 | 0.07438 | 1024/1024 | 30.93× | 0.06× | 0.91× |
| uni_G1024_s64 | 1024 | 0.0887 | 123.8 | 39.16 | 1.627 | 1.858 | 0.09462 | 1024/1024 | 24.07× | 0.05× | 0.94× |
| uni_G1024_s128 | 1024 | 0.2383 | 146.1 | 45.59 | 3.061 | 3.447 | 0.2456 | 1024/1024 | 14.89× | 0.08× | 0.97× |
| uni_G1024_s256 | 1024 | 1.32 | 176.6 | 57.74 | 9.922 | 10.57 | 1.327 | 1024/1024 | 5.82× | 0.13× | 0.99× |
| uni_G4096_s16 | 4096 | 0.245 | 398.5 | 106.2 | 4.405 | 4.878 | 0.2702 | 4096/4096 | 24.11× | 0.06× | 0.91× |
| uni_G4096_s32 | 4096 | 0.2697 | 463.5 | 144.7 | 5.197 | 5.816 | 0.2944 | 4096/4096 | 27.84× | 0.05× | 0.92× |
| uni_G4096_s64 | 4096 | 0.3309 | 495.5 | 157.5 | 7.041 | 8.031 | 0.355 | 4096/4096 | 22.37× | 0.05× | 0.93× |
| uni_G4096_s128 | 4096 | 0.9083 | 584 | 183.2 | 12.74 | 14.31 | 0.9353 | 4096/4096 | 14.39× | 0.07× | 0.97× |
| moe_E8_prefill_z0.0_up | 8 | 1.094 | 5.191 | 3.795 | 2.744 | 2.681 | 1.102 | 8/8 | 1.38× | 0.40× | 0.99× |
| moe_E8_prefill_z0.0_down | 8 | 1.115 | 8.898 | 4.998 | 2.65 | 2.45 | 1.116 | 8/8 | 1.89× | 0.42× | 1.00× |
| moe_E64_prefill_z0.0_up | 64 | 1.58 | 44.08 | 21.93 | 6.311 | 6.193 | 1.577 | 64/64 | 3.47× | 0.25× | 1.00× |
| moe_E64_prefill_z0.0_down | 64 | 1.593 | 33.65 | 16.51 | 6.467 | 6.539 | 1.602 | 64/64 | 2.55× | 0.25× | 0.99× |
| moe_E128_prefill_z0.0_up | 128 | 0.8746 | 70.51 | 28.2 | 6.365 | 6.239 | 0.8803 | 128/128 | 4.43× | 0.14× | 0.99× |
| moe_E128_prefill_z0.0_down | 128 | 0.9036 | 41.1 | 19.04 | 6.679 | 7.19 | 0.913 | 128/128 | 2.85× | 0.14× | 0.99× |
| moe_E8_decode_z0.0_up | 8 | 0.1595 | 3.524 | 2.021 | 1.052 | 0.9851 | 0.1607 | 8/8 | 1.92× | 0.15× | 0.99× |
| moe_E8_decode_z0.0_down | 8 | 0.2606 | 7.335 | 3.238 | 1.55 | 1.266 | 0.2606 | 8/8 | 2.09× | 0.17× | 1.00× |
| moe_E64_decode_z0.0_up | 64 | 1.053 | 40.36 | 20.06 | 5.435 | 5.317 | 0.7887 | 64/64 | 3.69× | 0.19× | 1.33× |
| moe_E64_decode_z0.0_down | 64 | 1.054 | 30.3 | 14.71 | 5.428 | 5.515 | 0.8043 | 64/64 | 2.71× | 0.19× | 1.31× |
| moe_E128_decode_z0.0_up | 128 | 0.8671 | 66.45 | 25.61 | 5.917 | 5.806 | 0.8706 | 126/128 | 4.33× | 0.15× | 1.00× |
| moe_E128_decode_z0.0_down | 128 | 0.8674 | 38.74 | 17.52 | 6.048 | 6.573 | 0.8752 | 126/128 | 2.90× | 0.14× | 0.99× |
| moe_E8_prefill_z1.2_up | 8 | 1.056 | 5.248 | 3.729 | 2.726 | 2.684 | 1.065 | 8/8 | 1.37× | 0.39× | 0.99× |
| moe_E8_prefill_z1.2_down | 8 | 1.117 | 8.737 | 4.884 | 2.693 | 2.439 | 1.117 | 8/8 | 1.81× | 0.41× | 1.00× |
| moe_E64_prefill_z1.2_up | 64 | 1.57 | 43.26 | 22.47 | 6.824 | 6.739 | 1.585 | 64/64 | 3.29× | 0.23× | 0.99× |
| moe_E64_prefill_z1.2_down | 64 | 1.595 | 33.14 | 17.31 | 6.969 | 7.083 | 1.609 | 64/64 | 2.48× | 0.23× | 0.99× |
| moe_E128_prefill_z1.2_up | 128 | 1.077 | 69.52 | 27.44 | 6.59 | 6.508 | 1.086 | 128/128 | 4.16× | 0.16× | 0.99× |
| moe_E128_prefill_z1.2_down | 128 | 1.118 | 40.77 | 18.83 | 7.098 | 7.639 | 1.129 | 128/128 | 2.65× | 0.16× | 0.99× |
| moe_E8_decode_z1.2_up | 8 | 0.16 | 3.511 | 2.011 | 1.068 | 0.9834 | 0.1608 | 8/8 | 1.88× | 0.15× | 1.00× |
| moe_E8_decode_z1.2_down | 8 | 0.2597 | 7.323 | 3.189 | 1.562 | 1.283 | 0.2608 | 8/8 | 2.04× | 0.17× | 1.00× |
| moe_E64_decode_z1.2_up | 64 | 1.052 | 32.29 | 16.11 | 4.436 | 4.274 | 0.6507 | 51/64 | 3.63× | 0.24× | 1.62× |
| moe_E64_decode_z1.2_down | 64 | 1.052 | 24.2 | 11.75 | 4.395 | 4.442 | 0.6602 | 51/64 | 2.67× | 0.24× | 1.59× |
| moe_E128_decode_z1.2_up | 128 | 1.048 | 52.22 | 20.14 | 4.775 | 4.609 | 0.7299 | 99/128 | 4.22× | 0.22× | 1.44× |
| moe_E128_decode_z1.2_down | 128 | 1.051 | 30.43 | 13.76 | 4.837 | 5.22 | 0.6873 | 99/128 | 2.84× | 0.22× | 1.53× |
| het_G64_16-2048 | 64 | 0.464 | 13.63 | 5.435 | 2.537 | 2.59 | 0.4715 | 64/64 | 2.14× | 0.18× | 0.98× |
| het_G256_16-2048 | 256 | 1.982 | 58.95 | 23.93 | 10.35 | 11.48 | 2.004 | 256/256 | 2.31× | 0.19× | 0.99× |
| lora_G16_r16_shrink | 16 | 0.1597 | 13.97 | 4.685 | 1.124 | 1.562 | 0.1618 | 16/16 | 4.17× | 0.14× | 0.99× |
| lora_G16_r16_expand | 16 | 0.056 | 2.882 | 1.617 | 1.131 | 1.31 | 0.05606 | 16/16 | 1.43× | 0.05× | 1.00× |
| lora_G64_rmix_shrink | 64 | 0.1654 | 54.3 | 17.38 | 1.737 | 1.735 | 0.1715 | 64/64 | 10.01× | 0.10× | 0.96× |
| lora_G64_rmix_expand | 64 | 0.1183 | 9.457 | 4.479 | 2.385 | 3.089 | 0.1233 | 64/64 | 1.88× | 0.05× | 0.96× |

**Geomean**（`tables.md` 表 2b、2c，各家族括號內是 workload 數）：

| 比較 | uniform (14) | moe (24) | hetero (2) | lora (4) | all (44) |
|---|---|---|---|---|---|
| emu_grouped 相對 emu_loop_batched（**模擬的加速**） | 18.49× | 2.65× | 2.23× | 3.25× | **4.97×** |
| emu_loop 相對 emu_loop_batched | 0.31× | 0.49× | 0.40× | 0.41× | 0.41× |
| emu_grouped 相對 native_grouped | 0.07× | 0.21× | 0.19× | 0.08× | **0.14×** |
| emu_grouped_reuse 相對 native_grouped | 0.07× | 0.22× | 0.18× | 0.06× | 0.13× |
| auto 相對 native_grouped（44 列都是 all-native，fallback 比例見上表） | 0.94× | 1.10× | 0.99× | 0.98× | 1.03× |
| native_loop 相對 native_grouped | 0.02× | 1.01× | 0.67× | 0.41× | 0.28× |

讀法：
- emu_loop → emu_loop_batched 是題內 batching 的增益；emu_loop_batched → emu_grouped 是跨題合併的增益，這是本作業的核心。**模擬的進步只看這兩步**，三者都是純模擬（COUNT）。
- 表 2c 中 auto 相對 emu_loop_batched 是 37.55×（geomean，tables.md 標為 all-native n = 44）。但 AUTO 的 fallback 是 100% 的非空問題，所以這個數字量到的是 cuBLAS 對模擬，**不能**當成 planner 或模擬的加速。tables.md 中 AUTO「emu」那一組（AUTO 至少模擬一題的列）是 n = 0。
- 6 列 MoE decode 的 auto fallback 小於 G（126/128、51/64、99/128）。差額是 m_e = 0 的空問題：native_loop 的 launch 數同樣是 126、51、99（表 3）。所以**沒有任何非空問題被模擬**。
- **量測假象**：上表中 moe_E64_decode 的 4 列與 moe_E128_decode_z1.2 的 2 列，native_grouped（1.048–1.054 ms）被高估了。原因是 bench 的路徑順序先量 native_loop，native_grouped 不在第一個位置（§10.8）。同樣排在第二的 moe_E128_decode_z0.0 兩列（0.8671、0.8674 ms）沒有被高估，所以這個效應不是每一列都會出現。所以這 6 列「auto 相對 NG」的 1.31×–1.62× 是假象，**不是** AUTO 比 native 快。第一個量測位置的 native_grouped（job 590956）與 AUTO 相同，例如 moe_E64_decode_z1.2_up 是 0.644–0.650 ms，AUTO 是 0.6507 ms。這 6 列的「EG 相對 NG」也應該更低，例如 moe_E64_decode_z1.2_up 是 0.6507/4.436 ≈ 0.15×（以 AUTO 時間代替公平的 native），不是表中的 0.24×。
- 590147 與 590258 的 launch 數逐列相同（308 列，由兩份 `bench.csv` 比對）。

### 7.2 Launch 數（`tables.md` 表 3，`ggemm_stats_t.kernel_launches`）

- emu_grouped 和 emu_grouped_reuse 在 44 列都是 3；auto 都是 1（一次 `cublasDgemmGroupedBatched`）；native_grouped 都是 1。
- emu_loop_batched 是 3 × 非空題數，例如 uni_G4096 是 12288。
- emu_loop 是 (N+3) × 非空題數，例如 uni_G4096_s32 是 73728，het_G256 是 4580（N 依 k 為 14–15）。
- native_loop 等於非空題數。
- AUTO 在 44 列的 moduli 都是 0-0，代表沒有任何一題被模擬。

### 7.3 應用層 E2E（job 590258 `tables.md` 表 4，ms）

**MoE FFN（FP64）**（`moe_ffn_fp64.csv`，median of 10）：

| config | native_grouped | emu_loop_batched | emu_grouped | emu_grouped_reuse | auto（fallback） | EG 相對 ELB | EG 相對 NG |
|---|---|---|---|---|---|---|---|
| T=4096 E=16 top2 1024/2048 zipf 0 | 2.624 | 11.90 | 6.515 | 6.149 | 2.628（32/32） | 1.83× | 0.40× |
| 同上 zipf 1.0 | 2.666 | 11.73 | 6.655 | 6.285 | 2.668（32/32） | 1.76× | 0.40× |
| 同上 zipf 1.5 | 2.665 | 11.34 | 6.511 | 6.157 | 2.659（32/32） | 1.74× | 0.41× |
| T=16384 E=8 top1 256/256 zipf 1.2 | 0.3189 | 2.088 | 1.575 | 1.595 | 0.3199（16/16） | 1.33× | 0.20× |

err_y/bound：EMU 路徑是 4.382e-05–9.341e-04，native_grouped 最大 6.076e-05。gemm 佔 E2E 的比例：EMU 路徑 91.7%–98.9%。

**Block-sparse BSR × dense**（`block_sparse_fp64.csv`，median of 5）：

| config | G | native_grouped | emu_loop_batched | emu_grouped | emu_grouped_reuse | auto（fallback） | EG 相對 ELB | EG 相對 NG |
|---|---|---|---|---|---|---|---|---|
| block 1024 dz 128 | 93822 | 15.90 | **call_failed** | **call_failed** | **call_failed** | 16.55（93822/93822） | — | — |
| panel 1024 dz 128 | 1024 | 0.9814 | 78.27 | 5.289 | 5.485 | 0.9723（1024/1024） | 14.80× | 0.19× |
| block 512 tz 256 | 46894 | 8.889 | 1865 | 147.9 | 132.4 | 9.158（46894/46894） | 12.61× | 0.06× |
| panel 512 tz 256 | 512 | 3.156 | 83.56 | 9.898 | 9.820 | 3.153（512/512） | 8.44× | 0.32× |
| block 256 qz 256 | 23380 | 5.332 | 1004 | 85.77 | 76.54 | 5.518（23380/23380） | 11.71× | 0.06× |
| panel 256 qz 256 | 256 | 2.098 | 74.27 | 10.16 | 9.600 | 2.108（256/256） | 7.31× | 0.21× |

- G = 93822 的 block 模式中，所有 EMU 路徑都回傳錯誤（call_failed），因為超過每次呼叫 65535 個模擬問題的上限（§10.7）。AUTO 全部 fallback，所以能執行。
- block 模式中 AUTO 全部 fallback 後，仍比 native_grouped 慢 3%–4%：16.55 對 15.90、9.158 對 8.889、5.518 對 5.332 ms。修正前（590147）分別是 11.67 對 8.846 ms 與 7.981 對 5.349 ms，shape memo 已大幅縮小這段差距（§9.6）。

### 7.4 Operand reuse

- **時間**：依 §6.2，以 bench 的 `emu_grouped_reuse` 列為準（job 590258 表 2a）。bench 的 44 個 workload 都沒有設計共用 operand（MoE／LoRA 每個 expert 或 adapter 的權重不同），所以量到的是 reuse 偵測的開銷。
  - 44 列中有 27 列 reuse 比 emu_grouped 慢。
  - emu_grouped_reuse/emu_grouped 的時間比介於 0.817–1.389，geomean 1.042，也就是平均慢 4.2%（由 `bench.csv` 計算）。
  - 應用 BSR 的 block 模式中 reuse 較快：132.4 對 147.9 ms、76.54 對 85.77 ms（表 4）。原因不在本次量測範圍內，不做推測。
- **正確性**：`test_reuse_e2e` 中 REUSE 和 EMU_GROUPED bit-exact，54/54 通過（job 590258）。
- `test_reuse_e2e` 印出的時間**不引用**（§6.2）。因此真正有共用 operand 時 reuse 的效能**目前沒有可引用的量測**。

### 7.5 CUDA Graph（Graph replay；job 590258 `logs/test_graph.log`，R = 200；AUTO 在這三個 workload 都全部 fallback，見 log 中 T1 的 `fallback=64/32/14`）

| workload | path | direct gpu_us | graph gpu_us | direct host_issue | graph host_issue | make_plan_ms |
|---|---|---|---|---|---|---|
| tiny64×(32,32,64) | EMU_GROUPED | 86.53 | 68.90 | 85.92 | 1.96 | 0.097 |
| tiny64×(32,32,64) | AUTO | 17.48 | 11.17 | 17.37 | 9.68 | 0.060 |
| moe32 | EMU_GROUPED | 612.89 | 600.39 | 606.97 | 1.95 | 0.111 |
| moe32 | AUTO | 100.20 | 100.37 | 99.26 | 86.48 | 0.089 |
| mixed16 | EMU_GROUPED | 217.47 | 204.04 | 215.39 | 1.96 | 0.078 |
| mixed16 | AUTO | 108.37 | 73.78 | 108.34 | 1.95 | 0.184 |

- 問題越小，graph 的收益越大：tiny 是 86.53/68.90 = 1.26×，moe32 只有 612.89/600.39 = 1.021×。
- graph 把 host issue 時間從 85.92 µs 壓到 1.96 µs，顯示 direct 呼叫的 host 規劃是可觀的固定成本。
- moe32 AUTO 用 graph 反而沒變快，graph 的 host_issue 仍有 86.48 µs。原因不在本次量測範圍內，不做推測。

---

## 8. Profiling 證據與瓶頸分析

### 8.1 nsys（job 511064，commit da18425；早於 prep split-K，最終版的結果見 §8.4）

資料來自 `results/prof_511064/analysis/nsys_summary_*.md`，皆為 5 次 timed call 的 median。

`uni_G1024_s32`：

| path | kernels/call | GPU span (µs) | gap % | host plan (µs) | host total (µs) |
|---|---|---|---|---|---|
| native_loop | 1024 | 10,720 | 80.7 | 0.0 | 10,731 |
| native_grouped | 2 | 31.7 | 26.3 | 0.0 | 58.3 |
| emu_loop | 18,432 | 113,365 | 15.7 | 5794.7 | 112,905 |
| emu_loop_batched | 3072 | 36,158 | 8.4 | 800.6 | 24,977 |
| emu_grouped | 3 | 577.5 | 1.0 | 675.7 | 702.8 |
| auto | 2 | 31.9 | 26.8 | 152.7 | 171.5 |

emu_grouped 的 GPU busy 時間按 stage 分：

| workload | prep % | INT8 GEMM % | reconstruct % |
|---|---|---|---|
| uni_G1024_s32 | 30.4 | 52.8 | 10.1 |
| moe_E64_prefill_z1.2_up | 63.4 | 29.6 | 6.9 |
| het_G256_16-2048 | 39.1 | 36.9 | 23.9 |
| lora_G64_rmix_shrink | 86.5 | 12.7 | 0.5 |

由此得到三個結論：
1. **launch 合併有效**：kernel 數從 18,432 降到 3，gap 從 15.7% 降到 1.0%。
2. **小問題時 host 規劃是瓶頸**：plan 時間 675.7 µs 大於 GPU span 577.5 µs，和 §7.5 graph 的觀察一致。
3. **較大的問題時，prep 是最大的 GPU stage**（63.4%–86.5%），所以最終版加入了 prep split-K。最終版的 profile（§8.4，job 590258）顯示 MoE up 的 prep 佔比仍是 63.4%，split-K 對 emu_grouped 的 prep 佔比幾乎沒有改變。

### 8.2 ncu（job 511064，emu_grouped，`--set full`，2 次 launch 的 median）

| workload | kernel | duration (µs) | INT8 pipe % | DRAM % | SM % | achieved occ % | 首要 stall |
|---|---|---|---|---|---|---|---|
| moe up | i8_grouped_persistent | 2450.3 | 44.5 | 31.6 | 48.8 | 24.9 | wait 29% |
| moe up | oz2_prep | 5521.0 | 0.0 | 23.6 | 70.0 | 60.6 | not_selected 29% |
| moe up | oz2_reconstruct | 548.1 | 0.0 | 23.6 | 58.5 | 46.9 | long_scoreboard 46% |
| het G256 | i8_grouped_persistent | 4195.1 | 23.3 | 23.8 | 35.4 | 25.0 | wait 24% |
| het G256 | oz2_prep | 4788.3 | 0.0 | 13.6 | 51.9 | 58.5 | short_scoreboard 28% |
| het G256 | oz2_reconstruct | 2744.1 | 0.0 | 25.2 | 59.8 | 47.5 | long_scoreboard 46% |
| lora shrink | i8_grouped_persistent | 335.8 | 48.0 | 17.3 | 47.1 | 24.0 | wait 30% |
| lora shrink | oz2_prep | 2055.0 | 0.0 | 5.3 | 18.5 | 27.4 | wait 36% |
| uni G1024 s32 | i8_grouped_persistent | 370.6 | 4.9 | 4.5 | 39.5 | 24.9 | long_scoreboard 35% |
| uni G1024 s32 | oz2_prep | 228.7 | 0.0 | 2.5 | 66.6 | 59.3 | short_scoreboard 31% |

- INT8 tensor pipe 最高只有 48.0%。kernel 用 128 registers、49,280 B smem，每 SM 只能放 2 個 CTA，occupancy 約 25%。
- LoRA shrink 的 prep：DRAM 5.3%、SM 18.5%，主要在 wait。這代表平行度不足，正是 prep split-K 要解決的問題。
- **AUTO 的 ncu 結果要小心解讀**：這些 workload 的 AUTO 全部 fallback，所以 ncu 量到的是 **cuBLAS 的 FP64 kernel**（job 511064 uni 的 AUTO 是 cuBLAS `Kernel2` 29.4 µs），不是我們的模擬 kernel。job 511064 中另外三個 AUTO 的 ncu log 出現 `LookupError: unknown encoding: utf-8-sig`，沒有產生摘要。

### 8.3 INT8 kernel 與 cuBLASLt（job 590258 `logs/test_i8_grouped.log`）

| set | ours（auto） | cuBLASLt-loop | 比值 |
|---|---|---|---|
| single 8192³ | 620.49 TOPS | 1470.04 TOPS | 42.2% |
| single 4096×4096×16384 | 610.20 | 1520.60 | 40.1% |
| ozaki-like 2048³×14 | 543.70 | 686.58 | 79.2% |
| random64 | 437.67 | 109.90 | ours 快 3.98× |
| mix 8×2048³ + 64 skinny | 422.34 | 54.61 | ours 快 7.73× |
| small256 | 0.0240 ms | 2.2635 ms | ours 快 94× |

在單一大 GEMM 上，我們的 kernel 只有 cuBLASLt 的約 40%；但在 grouped／異質的情況下，persistent + LPT 比逐題呼叫 cuBLASLt 快 4–94 倍。

### 8.4 最終版的 profiling（job 590258，`tables.md` §5 與 `analysis/nsys_summary_*.csv`）

**GPU-busy**（nsys，5 次 timed call 的 median，µs）：

| workload | path | kernels/call | GPU span | GPU busy | gap % | host (µs) | host plan (µs) | prep % | INT8 GEMM % | reconstruct % |
|---|---|---|---|---|---|---|---|---|---|---|
| uni_G1024_s32 | native_grouped | 2 | 32.48 | 23.71 | 27.6 | 54.39 | 0 | — | — | — |
| uni_G1024_s32 | emu_loop_batched | 3072 | 35,474 | 32,368 | 8.77 | 24,463 | 748.0 | 39.2 | 16.1 | 44.5 |
| uni_G1024_s32 | emu_grouped | 3 | 603.5 | 595.7 | 1.29 | 649.8 | 618.9 | 34.1 | 50.0 | 9.2 |
| uni_G1024_s32 | auto（fallback 1024/1024） | 2 | 31.81 | 23.52 | 26.1 | 60.47 | 42.0 | — | — | — |
| moe_E64_prefill_z1.2_up | native_grouped | 2 | 1585 | 1570 | 0.874 | 25.54 | 0 | — | — | — |
| moe_E64_prefill_z1.2_up | emu_loop_batched | 192 | 23,035 | 22,719 | 1.37 | 856.4 | 58.7 | 78.6 | 15.5 | 5.9 |
| moe_E64_prefill_z1.2_up | emu_grouped | 3 | 6802 | 6794 | 0.111 | 63.87 | 45.1 | 63.4 | 29.0 | 7.5 |
| moe_E64_prefill_z1.2_up | auto（fallback 64/64） | 2 | 1580 | 1569 | 0.685 | 27.17 | 9.7 | — | — | — |
| het_G256_16-2048 | native_grouped | 2 | 1985 | 1972 | 0.619 | 27.91 | 0 | — | — | — |
| het_G256_16-2048 | emu_loop_batched | 768 | 23,989 | 22,884 | 4.61 | 3488 | 229.1 | 52.5 | 24.9 | 22.5 |
| het_G256_16-2048 | emu_grouped | 3 | 10,201 | 10,194 | 0.0587 | 242.5 | 221.4 | 40.7 | 35.0 | 24.2 |
| het_G256_16-2048 | auto（fallback 256/256） | 2 | 1973 | 1961 | 0.633 | 61.62 | 41.5 | — | — | — |

（host plan 欄來自 csv 的 `plan_ns`；stage % 是 GPU busy 的佔比。）

- 在 uni_G1024_s32 上，emu_grouped 的 host（649.8 µs）仍然大於 GPU span（603.5 µs），host 規劃依然是瓶頸（host plan 618.9 µs）。
- AUTO 全部 fallback，GPU 時間和 native_grouped 相同（31.81 對 32.48 µs）。加入 shape memo 後，AUTO 的 host plan 從 152.0 µs（590147）降到 42.0 µs（590258），這就是 §10.1 regression 被修正的原因。

**ncu**（emu_grouped，kernel 2 次 launch 的 median）：

| workload | stage | duration (µs) | INT8 pipe % | DRAM % | SM % | occupancy % | regs | 首要 stall |
|---|---|---|---|---|---|---|---|---|
| het_G256 | i8_gemm | 4198 | 23.4 | 23.8 | 35.6 | 25.0 | 128 | wait 23.59% |
| het_G256 | prep | 5362 | 0 | 11.6 | 46.7 | 36.1 | 72 | short_scoreboard 33.48% |
| het_G256 | reconstruct | 2955 | 0 | 23.4 | 65.8 | 47.9 | 64 | long_scoreboard 42.56% |
| moe_E64 up | i8_gemm | 2456 | 44.5 | 31.5 | 48.8 | 24.9 | 128 | wait 29.47% |
| moe_E64 up | prep | 5571 | 0 | 22.7 | 66.8 | 36.8 | 72 | wait 35.03% |
| moe_E64 up | reconstruct | 612.2 | 0 | 21.1 | 65.1 | 47.4 | 64 | long_scoreboard 41.14% |
| uni_G1024_s32 | i8_gemm | 370.0 | 4.92 | 4.55 | 39.7 | 24.9 | 128 | long_scoreboard 35.25% |
| uni_G1024_s32 | prep | 267.3 | 0 | 2.19 | 58.2 | 36.2 | 72 | short_scoreboard 37.51% |
| uni_G1024_s32 | reconstruct | 66.22 | 0 | 23.3 | 53.2 | 45.1 | 64 | long_scoreboard 49.74% |

- 加入 split-K 後，prep 的 registers 從 48 增加到 72，achieved occupancy 從約 59–61% 降到約 36–37%（和 §8.2 比較）。
- 這個 workload 下，reconstruct 的 ncu duration 比 590147 略長（het 2955 對 2755 µs，moe 612.2 對 563.2 µs）；原因沒有量測，不做推測。計數器聚合的實測效益出現在 uncertified 很多的 repair sweep（§9.6）。
- 最終版 AUTO 的 ncu 摘要（`analysis/ncu_summary_*_auto.md`）沒有任何 kernel 列。這些 workload 的 AUTO 全部 fallback，就算有資料，量到的也會是 cuBLAS kernel，不是模擬。

**Roofline**（每條路徑在 44 個 workload 上達到自己 roof 的百分比）：

| path | points | median % | max % | bound |
|---|---|---|---|---|
| native_loop | 44 | 24.4 | 70.5 | compute 12、memory 32 |
| native_grouped | 44 | 36.2 | 49.4 | compute 12、memory 32 |
| emu_loop | 44 | 2.58 | 12.9 | memory 44 |
| emu_loop_batched | 44 | 5.75 | 18.0 | memory 44 |
| emu_grouped | 44 | 20.2 | 29.8 | memory 44 |
| emu_grouped_reuse | 44 | 19.6 | 28.0 | memory 44 |
| auto（all fallback） | 38 | 36.3 | 49.0 | compute 12、memory 26 |

emu_grouped 的 roof 利用率 median 是 20.2%，44 個點全部是 memory-bound。這和 prep 是最大 stage 的觀察一致。

---

## 9. Ablation

### 9.1 路徑階梯

| 步驟 | 比較 | 隔離出的機制 | 數字 |
|---|---|---|---|
| A | emu_loop → emu_loop_batched | 題內 batching | geomean 1/0.41 ≈ 2.44×（表 2c 的 emu_loop 欄是 0.41×；各家族 0.30–0.49×） |
| B | emu_loop_batched → emu_grouped | 跨題合併（3 kernel + LPT） | geomean **4.97×**；範圍 1.37×（moe_E8_prefill_z1.2_up）到 30.93×（uni_G1024_s32）（job 590258 表 2c） |
| C | emu_grouped → auto | planner | **無法評估**：AUTO 沒有模擬任何非空問題（§7.1），表 2c 中 auto 的 37.55× 是 cuBLAS 對模擬 |
| D | emu_grouped → emu_grouped_reuse | operand reuse | bench 中沒有共用 operand 的 workload：geomean 慢 4.2%（§7.4） |
| E | direct → graph | CUDA Graph | EMU_GROUPED 的 Graph replay 是 1.021–1.26×（§7.5） |
| F | COUNT → ON | repair 成本 | N = 15：φ ≤ 1 時 0.990–1.004；φ = 2 時 1.186–4.113；全表最大 5.634（job 590257，§5.3） |
| G | planner 無 memo → 有 memo | per-call shape memo | AUTO 全部 fallback 時的開銷：uni_G4096_s32 從 0.7375 降到 0.2944 ms（§9.6） |
| H | 每 thread atomic → warp+block 聚合 | 計數器聚合 | repair sweep EMU 192 列，時間 geomean 0.959×（§9.6） |

### 9.2 INT8 kernel 第二輪改動的 ablation（負面結果；job 510965，pass 2，單位 TOPS；small256 那欄是 ms）

| variant | 內容 | random64 cfg0 | ozaki cfg0 | 8192³ cfg0 | 4096×4096×16384 cfg0 | small256 cfg2 |
|---|---|---|---|---|---|---|
| r1 | 第一輪 | 440.17 | 543.40 | 585.84 | 607.59 | 0.0456 |
| r2 | 第二輪 | 391.91 | 481.55 | 534.67 | 536.23 | 0.0725 |
| no_po | 拿掉 prologue overlap | 399.00 | 493.69 | 534.36 | 537.92 | 0.0695 |
| zfill | 加 zero-fill | 387.89 | 479.00 | 527.41 | 524.21 | 0.0732 |
| dec_end | decode 移到最後 | 375.31 | 457.22 | 518.26 | 515.59 | 0.0784 |
| dec_every4 | 每 4 步 decode 一次 | 401.61 | 499.13 | 542.71 | 549.78 | 0.0757 |
| no_splitk | 拿掉 split-K | 407.74 | 506.90 | 548.09 | 558.00 | 0.0746 |
| minb1 | 1 CTA/SM，186 regs | 302.34 | 376.62 | 424.70 | 430.43 | 0.1213 |
| revert_all | 四項全部還原 | 402.95 | 494.43 | 555.36 | 517.71 | 0.0714 |

r2 的 cfg0 退步 8.7%–11.7%。把四項改動全部還原後仍回不到 r1，表示退步是結構性的。所以最終版採用 r1 加上幾個安全修正。

### 9.3 Split-K

**同一個 kernel 內做 split-K**（job 511001，pass 2，TOPS）：

| workload | cfg2 off | cfg2 splitk | cfg0 off | cfg0 splitk |
|---|---|---|---|---|
| random64 | 128.66 | 177.60 | 440.18 | 423.47 |
| ozaki | 141.29 | 194.95 | 543.80 | 521.07 |
| single 8192 | 164.69 | 216.92 | 585.15 | 569.41 |
| single 4096 | 173.61 | 265.68 | 599.80 | 582.63 |

cfg2 提升 31.7%–53.0%，但 cfg0 下降 2.7%–4.2%，所以不採用。

**獨立的 split-K kernel（cfg3）**（job 511054）：small256 從 auto 的 0.0236 ms 變成 auto_sk 的 0.0310／0.0313 ms，慢約 31%；mix 從 422.79 降到 409.59 TOPS，慢約 3%。因此預設關閉，而且經由 ggemm API 無法啟用（§2.3）。最終版的 job 590258 結果一致：small256 auto 0.0240 ms、auto-sk 0.0314 ms。

### 9.4 Tile config（job 590258，TOPS；small256 為 ms）

| set | cfg0 | cfg1 | cfg2 | auto |
|---|---|---|---|---|
| random64 | 441.26 | 290.05 | 128.97 | 437.67 |
| mix | 407.06 | 289.54 | 134.89 | 422.34 |
| single 8192³ | 598.17 | 368.53 | 165.91 | 620.49 |
| small256 (ms) | 0.0266 | 0.0260 | 0.0460 | 0.0240 |

在異質的情況（mix、small256）下，auto 優於任何單一固定 config；random64 上 auto 和 cfg0 相差 0.9% 以內。

### 9.5 Repair mode 與 ρ_budget（job 590257；舊版 590148 結論相同）
- **Repair mode**：見 §5.3 與 §9.1 步驟 F。
- **ρ_budget ∈ {4, 16, 64, 256}**：四個 repair workload 的 AUTO 在所有 ρ 與 φ 下都全部 fallback。moduli 都是 [0,0]，fallback 等於 G（uni_G1024_s64 是 1024；uni_G256_s256、het 是 256；moe 是 64），repaired 比例都是 0。時間不隨 ρ 變化，例如 uni_G1024_s64 是 0.09488–0.09670 ms（590148 時是 0.1957–0.2033 ms，差別來自 shape memo）。所以**在 H200 上無法評估 ρ_budget**，預設值至今仍沒有調過。
- AUTO 相對 native_loop 的時間：uni_G1024_s64 φ = 0.5、ρ = 16 是 0.09664 對 7.2516 ms；het 是 2.0113 對 3.0567 ms。這反映的是 native_grouped 對 native_loop 的差距。

### 9.6 兩項修正的前後對照（590147 → 590224 → 590258；590148 → 590257）

**(a) Planner per-call shape memo**（commit a48eb3e；§2.6）

| workload | AUTO：590147（b56631d） | AUTO：590224（a48eb3e） | AUTO：590258（1c39e40） | native_grouped（590258） |
|---|---|---|---|---|
| uni_G4096_s32 | 0.7375 ms | 0.2967 ms | 0.2944 ms | 0.2697 ms |
| uni_G1024_s32 | 0.1845 ms | 0.0746 ms | 0.07438 ms | 0.06762 ms |

- AUTO 比 native_grouped 慢 5% 以上的列數，從 590147 的 14/44 降到 590258 的 9/44（由 `bench.csv` 計算）。剩下的 9 列都是小問題的 uniform workload，慢 6.7%–10.3%，例如 uni_G4096_s16 0.2702 對 0.245 ms、uni_G1024_s32 0.07438 對 0.06762 ms。這是剩餘的 host plan 開銷：nsys 量到 uni_G1024_s32 的 AUTO host plan 是 42.0 µs（§8.4）。
- 正確性：planner 單元測試 1655/0（job 590223），包含 memo 與非 memo 的 plan 逐 entry bit-identical 比對（`tests/test_planner.cpp` 第 403 行起）；plan 呼叫中的 `operator new` 次數為 0。

**(b) Reconstruct 計數器改成 warp + block 聚合**（commit 1c39e40；§2.5）

- **計數不變**：repair sweep 舊版 590148 與新版 590257 的 EMU 結果共 192 列，以 (workload, path, G, φ, N, repair_mode, ρ, total_entries) 配對後，num_uncertified、num_repaired、num_safety_fallback、n_over_gamma、err_max **全部完全相同**。連同 AUTO／native 的 128 列，320 列都相同（由兩份 CSV 比對）。
- **時間**：EMU 192 列的時間比（新/舊）geomean 是 0.959；18 列快了 20% 以上，8 列慢了 5% 以上（最大 1.248）。只看 COUNT 模式的 geomean 是 0.918。效果最明顯的是 uncertified 很多的格子：
  - het_G256 φ = 2 N = 13 COUNT：24.68 → 9.358 ms。
  - uni_G256_s256 φ = 2 N = 13 COUNT：5.148 → 2.261 ms。
- 這證實了 `docs/qa_prep.md` 先前的推論：舊版每個 thread 都對同一個全域位址做 atomicAdd，這些 atomic 會被序列化。
- 副作用：COUNT 變快以後，ON/COUNT 的比值變大（§5.3）。repair 的絕對成本沒有變，只是不再被 atomic 的開銷掩蓋。

---

## 10. 失敗案例與負面結果

### 10.1 H200 上 native FP64 全面勝出，AUTO 全部走 native，而且 AUTO 有開銷 regression

**正式版**（job 590258，表 2a、2b、3）：
- emu_grouped 在 **44/44** 列都比 native_grouped 慢，慢 2.38×（moe_E8_prefill_z0.0_down）到 21.3×（uni_G4096_s64），geomean 0.14×。這是以 bench 矩陣中的 native_grouped 計算的；其中 6 列 MoE decode 的 native_grouped 被量測順序高估了（§10.8），用公平的 native baseline 只會讓模擬**更**不利。例如 moe_E64_decode_z1.2_up 從表中的 4.22×（4.436 / 1.052）變成約 6.8×（4.436 / 0.650）。
- AUTO 有 38/44 列 fallback = G；其餘 6 列 fallback = 非空題數；44 列的 `moduli_max` 都是 0。**AUTO 在整個 bench 中沒有模擬任何一題**。repair sweep 的 ρ_budget 掃描也一樣全部 fallback（§9.5）。
- **AUTO 全部 fallback 時的開銷**：
  - 修正前（job 590147）有 14/44 列 AUTO 比 native_grouped 慢 5% 以上，例如 uni_G4096_s32 是 0.7375 對 0.2727 ms（2.7 倍）、uni_G1024_s32 是 0.1845 對 0.06827 ms（2.7 倍）。
  - 加入 planner shape memo 後（job 590258）降到 **9/44 列**，例如 uni_G4096_s32 是 0.2944 對 0.2697 ms、uni_G1024_s32 是 0.07438 對 0.06762 ms。剩下的差距是 6.7%–10.3%，全部在小問題的 uniform workload（§9.6）。uniform 家族的 geomean 從 0.63× 改善到 0.94×。
  - **殘留的 regression 照實保留**：在小問題的 uniform 上，AUTO 仍然比直接呼叫 native_grouped 慢約 7%–10%。BSR 的 block 模式也有一樣的情形（慢 3%–4%，§7.3）。

前一版 commit da18425 的 quick set（job 511065 `bench.csv`，φ = 0.5，median of 20，ms），結論相同：

| workload | native_grouped | emu_grouped | auto | auto fallback |
|---|---|---|---|---|
| uni_G1024_s32 | 0.0735 | 1.2533 | 0.1855 | 1024/1024 |
| moe_E64_prefill_z1.2_up | 1.5684 | 6.7115 | 1.5762 | 64/64 |
| moe_E64_prefill_z1.2_down | 1.5887 | 6.8264 | 1.5970 | 64/64 |
| het_G256_16-2048 | 1.9709 | 9.7146 | 2.0079 | 256/256 |
| lora_G64_rmix_shrink | 0.1658 | 2.1083 | 0.1723 | 64/64 |
| lora_G64_rmix_expand | 0.1176 | 2.1440 | 0.1214 | 64/64 |

模擬比 native 慢的原因有三：
1. H200 的 FP64 很強，擬合值是 61.3 TFLOPS；模擬卻要做約 N+1 = 16 份 INT8 GEMM，外加 prep 和 reconstruct。
2. 我們的 INT8 kernel 只有 cuBLASLt 的約 40%。
3. prep 是主要成本。

planner 把問題路由到 native 的判斷是正確的，但它本身的 host 開銷（修正後仍有，例如 uni_G1024_s32 的 host plan 是 42.0 µs）讓 AUTO 在小問題上略輸給直接呼叫 native_grouped（見上）。FP64 較弱的硬體上結果是否會反轉，我們**沒有實測**。校正檔裡的 `crossover_square_pow2_by_N`（N = 15 時方陣邊長 ≥ 4096 才划算）只是 model 推估，沒有被驗證；而且它用的是 cuBLASLt 的 INT8 峰值，對我們的 kernel 偏樂觀。

### 10.2 INT8 kernel 的效率缺口
在單一大 GEMM 上只有 cuBLASLt 的 40.1%–42.2%（job 590258），tensor pipe 最高 48.0%，occupancy 約 25%，第二輪改動還退步約 10%（§8.2、§8.3、§9.2）。

### 10.3 compute-sanitizer

**第一次執行（job 511062，commit da18425）**：

| 程式 | memcheck | racecheck | initcheck | synccheck |
|---|---|---|---|---|
| test_grouped | 0 | log 是 0 hazards（summary.md 誤標成 no_summary） | 0 | 0 |
| test_reuse_e2e | 0 | TIMEOUT（900 s） | 0 | 0 |
| test_graph | 0 | TIMEOUT（900 s） | 0 | 0 |
| test_i8_grouped | 3 個 leak，共 67,174,608 bytes（測試程式 cudaMalloc 後沒 free） | log 是 0 hazards | 19,090 筆（summary.md 寫成 18990）：`check_vs_lt` 用 cudaMemcpy 讀回未初始化的 buffer（是測試 harness 的問題） | 0 |
| test_operand_reuse | host-only 程式，sanitizer rc = 255，但程式本身 `all passed` | n/a | n/a | n/a |

job 511062 回報的問題都出在測試程式或摘要腳本，不是 library 的 kernel：
- **initcheck**：`test_i8_grouped` 的 harness 用 `cudaMemcpy` 把 device buffer 中從未寫入的 padding 一起複製回 host，所以被判定為讀取未初始化記憶體。log 的 `ERROR SUMMARY` 是 **19,090** 筆，summary.md 寫的 18990 其實是「未印出」的筆數（log 中的 `18990 errors were not printed`），兩者不同。
- **memcheck**：測試程式有 3 塊 `cudaMalloc` 沒有釋放，共 67,174,608 bytes 的 leak。
- **racecheck**：test_grouped 與 test_i8_grouped 的 log 都是 `0 hazards`，但摘要腳本的解析有 bug，誤標成 no_summary。test_reuse_e2e 與 test_graph 是全尺寸跑到 timeout。

**重跑**（job 590161，commit b56631d，dirty=8：只修改 tests/ 與 scripts/，`src/` 未動。修正內容：harness 只複製寫入過的範圍並釋放記憶體、修好 racecheck 摘要解析、新增 `--small` 模式、host-only 程式標成 n/a；`results/sanitize_590161/summary.md`）：

| 程式 | memcheck | racecheck | initcheck | synccheck | plain 執行 |
|---|---|---|---|---|---|
| test_grouped | 0 | 0 | 0 | 0 | — |
| test_reuse_e2e | 0 | 0 | 0 | 0 | — |
| test_graph | 0 | 0 | 0 | 0 | — |
| test_i8_grouped | 0 | 0 | 0 | 0 | — |
| test_operand_reuse | n/a（host-only） | n/a | n/a | n/a | 0 |

四個 GPU 測試程式在四種 sanitizer 工具下都是 0 errors。590161 跑的是 b56631d 的 `src/`；之後 a48eb3e 與 1c39e40 改過 src，因此在最終程式碼上再跑一次：job 590498（commit a6b337d，`src/` 與 1c39e40 相同，dirty=4 只有文件），結果相同——四個 GPU 測試在 memcheck／racecheck／initcheck／synccheck 全部 0 errors，test_operand_reuse 為 host-only（n/a），直接執行 0 錯（`results/sanitize_590498/summary.md`）。

### 10.4 Operand reuse 的增益沒有被證明
- 正式版 bench（590258）中有 27/44 列 reuse 比 emu_grouped 慢，geomean 慢 4.2%（§7.4）。例如 uni_G4096_s64 是 8.031 對 7.041 ms，lora_G64_rmix_expand 是 3.089 對 2.385 ms。
- nsys 顯示 reuse 的 prep 佔比較高：het 是 46.6% 對 40.7%，GPU span 11,330 對 10,200 µs（`tables.md` §5）。
- bench 裡沒有共用 operand 的 workload，而有共用 operand 的 `test_reuse_e2e` 計時不可引用（§6.2）。所以 **reuse 帶來的效能增益目前沒有可引用的證據**。
- 早先的影片（10/07）引用的 reuse 1.015–1.079×（job 510998）來自 `test_reuse_e2e`，而且當時對照組（不共用）也有同樣程度的加速，因此作廢。

### 10.5 寬動態範圍的代價
- 純模擬（COUNT）的 err/gamma 最高到 175.9（test set，job 590258），repair sweep 中最高到 1.018e+04（job 590257、590148）。
- 開 ON 之後全部 ≤ 0.4559，代價是 φ = 2、N = 15 時慢 1.19–4.11 倍，最壞情況慢 5.63 倍（job 590257）。
- ON 模式平均有 13.3% 的 entry 要用 FP64 重算；N = 13 時部分 workload 超過 99.9%（§5.3）。
- **repair ON 是正確性的安全網，不是加速**。

### 10.5b 10/07 進度影片的數字已過時
影片（`docs/video/slides.json`）中的 LoRA 8.4×、TP 5.2×、none 6.4×（EMU_LOOP_BATCHED → EMU_GROUPED，job 510998）和 reuse 1.015–1.079× 都是舊 commit 的 `test_reuse_e2e` 計時：
- 這個計時內含同步，依 §6.2 不引用。
- 最終 commit 的 prep split-K 讓 EMU_LOOP_BATCHED 大幅變快（例如 bench 的 lora_G16_r16_shrink，ELB/EG 現在是 4.17×）。

本報告一律以 job 590258 為準。另外，影片中「完整正確性測試」引用的是 job 511059；最終版本的對應數字是 job 590256、590258（都是 2955/0）。

### 10.5c 量測與文件本身的缺陷
- 校正檔的 INT8 擬合峰值 1676 TOPS 高於同一檔案中的實測最大值 1503.6 TOPS（`i8_peak_ops` 對 `i8_peak_ops_max`）。兩者都來自 cuBLASLt，而我們的 kernel 只有 cuBLASLt 的約 40%，所以 cost model 偏樂觀。這不影響 H200 上「全部 fallback」的結論，但會影響 crossover 的推估（§10.1）。
- job 590148 與 590257 的 ρ_budget 掃描全部 fallback，無法提供資訊，所以 ρ_budget 預設值仍未調過（§9.5）。
- 部分 results 沒有記錄 commit（511060、511063、511101），部分早期 job 是在 dirty tree 上跑的（510965 dirty=18、510989 dirty=36）。這些數字只用於歷史對照或 ablation，最終結論都基於 job 590258、590257、590256（commit 1c39e40；dirty 只涉及文件檔，`src/` 乾淨）。

### 10.6 開發過程中失敗的 job（保留在 `results/` 中）

| job | 失敗內容 | 後續 |
|---|---|---|
| 510881 | 2007 passed / **26 failed**：AUTO 全部 fallback 時 moduli 範圍是 [0,0] 觸發 assertion；φ = 2 的 het／lora 上 emu 的 err/gamma 為 12.5、24.3（當時還沒有 certify-and-repair） | 引入 certify-and-repair 與 cert gate |
| 510997 | 2487 / **1 failed**（zero_dims_mix 的 auto） | 修正後 511059、511249、590162 都是 2955/0 |
| 510961 | test_graph 55 / **8 failed**（EMU_GROUPED 不是 bit-exact） | 修正後 510998、511060、590147 都是 63/63；results 中沒有記錄根因 |
| 511002、511003 | `ggemm.cu` 編譯失敗（`oz2_read_counters` 介面不一致，6 個 error） | 對齊介面後重跑（511065、511066） |
| 511061 | repair sweep 出現 `ar: unable to copy file .../build/libggemm.a`；511061／511062／511064 在同一分鐘啟動，共用同一個 `build/` 目錄 | final runner 改為每個 job 用自己的 build 目錄；repair sweep 由 590148 重跑完成 |

### 10.8 量測假象：MoE decode 上 native_grouped 的時間取決於量測位置

**現象**（job 590956，`scripts/analysis/path_order.slurm`，`results/order_590956/summary.txt`；moe_E64_decode 4 個 workload，6 種路徑順序，每種跑 2 次）：

| 情況 | native_grouped（ms） |
|---|---|
| 只量 native_grouped | 0.644–0.860 |
| native_grouped 排第一（其後是其他路徑） | 0.645–0.864 |
| 同一個 process 連量兩次 native_grouped：第一次 | 0.644–0.867 |
| 同上：第二次 | 1.034–1.049（有一次是 0.849） |
| 排在其他路徑之後 | 多數是 1.027–1.049；少數仍快，例如 0.647、0.652、0.849、0.954 |
| AUTO（cuBLAS 呼叫參數相同），任何位置 | 0.647–0.871，不受影響 |

**排除過的原因與診斷**：
- v0 與最終版的 cuBLAS handle 設定相同（`cublasSetWorkspace`、`cublasSetStream`、`cublasSetPointerMode`、`cublasSetMathMode`）。
- job 590953（`results/ndiag_590953/`）把 native_grouped 的分組改回 v0 的「每題一個 size-1 group」，**沒有效果**：
  - moe_E64_decode 仍是約 1.04–1.05 ms，例如 z1.2_up 是 1.0484／1.0494／1.0476 ms。
  - moe_E128_decode 反而變慢，例如 z1.2_up 從 0.7277／0.7294／0.7265 ms 變成 0.8854／1.0475／1.0486 ms。
  - 所以這個修改**沒有採用**。
- 在 nsys 底下只跑 native_grouped 和 auto 兩條路徑時，兩者時間相同：moe_E64_decode_z1.2_up 是 0.659 對 0.662 ms（`nsys_base_bench.csv`）。
- 其他 workload 不受影響。590953 中 moe_E8 與 moe_E64 prefill 的數字在兩個版本都一致，例如 moe_E64_prefill_z1.2_up native_grouped 是 1.5729–1.5826 對 1.5753–1.5825 ms。
- **根因在 cuBLAS grouped GEMM 內部，未查明。**

**影響與修正**：
1. bench 矩陣（590258、590147）中 MoE decode 的 native_grouped 被高估約 1.6×（1.052 對 0.650 ms），只出現在 moe_E64_decode 的 4 列與 moe_E128_decode_z1.2 的 2 列。公平的 native baseline 用第一個量測位置的值（上表），或用同一列的 AUTO 時間（cuBLAS 呼叫參數相同、不受位置影響）。
2. 先前報告中「AUTO 在 MoE decode 比 native_grouped 快 1.31–1.62×」是這個假象，已在 §7.1 改寫。AUTO 與 native 實際上一樣快。
3. v0→final 的 native_grouped 0.95×（§7.0）也是同一個假象：受影響的 MoE decode 列 v0 是 0.65–0.82 ms，final 是 0.88–1.05 ms。**不是** regression，兩版呼叫 cuBLAS 的方式相同。
4. 改用公平的 native baseline 之後，模擬相對 native 的結論只會**更不利**（§10.1）。

### 10.7 其他已知限制（摘自 README）
- 不支援轉置。
- Kp 上限 131040。
- 一次呼叫最多 65535 個模擬問題。實際影響：BSR block 模式 G = 93822 時，所有 EMU 路徑都 call_failed（§7.3）。
- graph 路徑不合併 native fallback，也不做 reuse；plan 必須在 handle 之前 destroy。
- 不支援 multi-GPU。
- workspace 在 create 時固定；k < 32 時要用 `ggemm_create_ex` 的 `min_k`。
- ρ_budget 未調。
- 單一 stream，handle 與 plan 都不是 thread-safe。
- 只支援 FP64 實數。
- N 只由 (k, target) 決定，可能多配。

---

## 11. AI 與外部程式碼使用說明

**AI 工具。** 本專案全程使用 **Claude Code**（Anthropic），以多個 agent 分工。分工記錄在 `DESIGN.md` 的「模組與檔案歸屬」表，以及各原始檔檔頭的 `owner: agent-X`：

| agent | 負責 |
|---|---|
| N | numerics |
| K | INT8 kernel |
| P | pipeline／planner |
| B | tests／bench／scripts |
| G | CUDA Graph |
| E | 應用 bench |
| L | 文獻 |
| A | final runner |

AI 的用途：
- 閱讀論文與整理文獻：`docs/related_work.md` 的作者與 arXiv 編號已在 10/07 人工查證並修正，修正記錄在該檔內。
- 實作 library、測試、bench 與 SLURM 腳本。
- 除錯（§10.6 那些失敗的 job）。
- 分析 profiling 結果（`scripts/analysis/*.py`）。
- 撰寫文件、本報告的初稿、口試準備（`docs/qa_prep.md`），以及 10/07 進度影片的投影片、旁白和工具鏈（`docs/video/`，語音用 `edge_tts` 產生）。

**人類的角色。** 理解題目、做方向與取捨的決策、審查 agent 的產出、核准 SLURM job。所有 GPU 程式都在 compute node 上執行，login node 只用來編譯。

**Git 歷史。** 歷史完整保留，目前共 10 個 commit：
- 前 5 個（5d8e2ad、173bb94、da18425、1422998、b56631d）都在 2026-10-07 由 `moneychen168` 提交，粒度很粗。
- 之後依序是 bc310b8（sanitizer 誤報修正）、a48eb3e（planner shape memo、`host_allocs_timed`）、97cde81（qa_prep.md）、1c39e40（計數器聚合）、6ac461e（本報告初版）。

每個 results 目錄都有 `provenance.txt`，記錄當時的 hash 與 dirty 檔案數。部分早期 job 是在 dirty tree 上跑的（510887 dirty=3、510965 dirty=18、510989 dirty=36）。**正式量測 590258（以及 590256、590257）都在 commit 1c39e40**。provenance 記錄 dirty=6，`provenance_git.txt` 列出的是 4 個修改過的文件檔（DESIGN.md、README.md、docs/discussion_1007.md、docs/related_work.md）和 2 個未追蹤項目（docs/report.md、scripts/analysis/__pycache__/），`src/` 乾淨。舊版 590147、590148、511249 是 dirty=0（b56631d）。

**外部程式碼：沒有複製任何外部實作。**
- **GEMMul8**：只讀過它的論文和演算法，沒有使用其程式碼（`src/ozaki2_numerics.cu` 檔頭註明「Written from the paper; no code taken from GEMMul8」）。`EMU_LOOP_BATCHED` 是我們自己寫的 GEMMul8 風格 baseline。
- **CUTLASS**：沒有使用（`int8_grouped_gemm.cu` 檔頭註明「No CUTLASS」）。INT8 kernel 用 inline PTX 自行撰寫。
- **cuBLAS**：只用在 native 路徑、fallback、校正與比較。repair 用的是我們自己寫的 FP64 FMA 內積。
- **cuBLASLt**：只用來當 INT8 的參考與比較（`i8_gemm_cublaslt_single`、`test_i8_grouped`、`calib.cu`）。
- **NVTX**：CUDA toolkit 附的 header-only nvtx3。
- **演算法依據**：arXiv:2504.08009（Ozaki scheme II）、Ozaki et al. 2012、Higham 2002 §3.5。

---

## 12. 重現方式

所有指令都在 repo 根目錄用 sbatch 送出，**不要在 login node 執行任何 GPU 程式**。腳本的預設參數是 `-A GOV114009 -p dev --gpus-per-node=1 --cpus-per-task=12`，並排除 `25a-hgpn001,25a-hgpn003,25a-hgpn085`。要換帳號就用 `sbatch -A <account>` 覆蓋。

```bash
mkdir -p results logs
sbatch scripts/final/run_final.slurm   # build→全部測試→bench full set→應用→nsys/ncu→分析→tables.md
#   產出 results/final_<jobid>/{summary.md,tables.md,steps.tsv,bench.csv,tests.csv,prof/,analysis/}
#   旋鈕：REPS=20 WARMUP=5 PHI=0.5 BENCH_SET=bench REPS_MOE=10 REPS_BSR=5 SKIP_NCU=1 SKIP_NSYS=1 REQUIRE_CLEAN=1
#   重新產生表格（只解析 CSV）：python3 -I scripts/final/aggregate.py results/final_<jobid>
sbatch scripts/repair_sweep.slurm      # COUNT/ON、ρ sweep → results/repair_<jobid>/plots/repair_summary.md
sbatch scripts/sanitize.slurm          # compute-sanitizer → results/sanitize_<jobid>/summary.md
sbatch scripts/run_tests.slurm         # test_grouped（含 memcheck）
sbatch scripts/run_planner_unit.slurm
sbatch scripts/run_bench.slurm
sbatch scripts/profile.slurm
sbatch bench/app/run_app.slurm
sbatch bench/calib/run_calib.slurm     # 重新校正 → bench/calib/params_<gpu>.json
sbatch scripts/i8_ablation.slurm; sbatch scripts/i8_cfg2.slurm; sbatch scripts/i8_cfg2sep.slurm; sbatch scripts/run_i8_unit.slurm
sbatch scripts/final/starter_vs_final.slurm  # starter（v0 = 5d8e2ad）與最終版，同一個 job → results/v0final_<jobid>/compare.md
sbatch scripts/analysis/path_order.slurm     # MoE decode 的量測順序效應 → results/order_<jobid>/summary.txt
```

正式版本的程式是 commit 1c39e40（見下面的版本說明）。核心程式碼逐段說明、profiling 的解讀，以及「改參數時的預測」整理在 `docs/qa_prep.md`（口試準備）。

**版本說明**：
- **正式版本是 commit 1c39e40，job 590258**（`results/final_590258/`，35 步全部 PASS），以 `sbatch scripts/final/run_final.slurm` 產生。
- starter 與最終版的比較是 590633（v0 5d8e2ad 對最終版 e20a669；e20a669 的 `src/`、`include/` 與 1c39e40 相同）。量測順序假象的診斷是 590956 與 590953（§10.8）。
- 修正前後的對照用：job 590147（b56631d，修正前）、590224（a48eb3e，只加 shape memo）（§9.6）。
- repair sweep 是 590257（1c39e40），對照 590148（b56631d）。
- 測試是 590256（1c39e40），planner 單元測試是 590223（a48eb3e），sanitizer 是 590161（b56631d，dirty=8，只修改了 tests／scripts）。之後在最終程式碼上重跑 sanitizer：job 590498（commit a6b337d，`src/` 與 1c39e40 相同），四個 GPU 測試在 memcheck／racecheck／initcheck／synccheck 全部 0，test_operand_reuse 為 host-only（n/a），直接執行 0 錯（§10.3）。

---

## 附錄 A：數字來源

| job | 種類 | commit（dirty） | 用於 | 檔案 |
|---|---|---|---|---|
| **590258** | **正式 final matrix（35 步全部 PASS）** | 1c39e40 (6，只有文件檔) | §0、§2、§3、§5、§6、§7、§8.3、§8.4、§9、§10 | `results/final_590258/tables.md`、`bench.csv` 等 |
| 590633 | starter（v0）與最終版比較 | v0 5d8e2ad；final e20a669（dirty=2，`src/` 與 1c39e40 相同） | §0、§2.9、§6.2、§7.0、§10.8 | `results/v0final_590633/compare.md` |
| 590956 | 路徑順序效應 | c8463fc (4) | §6.2、§7.1、§10.1、§10.8 | `results/order_590956/summary.txt` |
| 590953 | native_grouped 分組診斷 + nsys | c8463fc (5) | §10.8 | `results/ndiag_590953/summary.md`、`nsys_base_bench.csv` |
| 590256 | test_grouped | 1c39e40 (6) | §0、§5.2 | `results/test_590256/`、`slurm-ggemm-test-590256.out` |
| 590257 | repair sweep（正式） | 1c39e40 (6) | §5.3、§9.1、§9.5、§9.6、§10.5 | `results/repair_590257/plots/repair_summary.md`、`emu_repair.csv` |
| 590224 | final matrix（只加 shape memo，對照用） | a48eb3e (4) | §9.6 | `results/final_590224/bench.csv` |
| 590223 | planner unit | a48eb3e（log 中未記錄 dirty） | §2.6、§5.2、§9.6 | `results/planner_unit_590223/test_planner.log` |
| 590147 | final matrix（修正前，對照用） | b56631d (0) | §9.6、§10.1、§7.3（對照） | `results/final_590147/tables.md` |
| 590148 | repair sweep（修正前，對照用） | b56631d (0) | §5.3、§9.6 | `results/repair_590148/` |
| 590162 | test_grouped 重跑 | b56631d (8，只動 tests／scripts) | §5.2 | `results/test_590162/` |
| 590161 | sanitizer 重跑 | b56631d (8，只動 tests／scripts) | §10.3 | `results/sanitize_590161/summary.md` |
| 590498 | sanitizer（最終程式碼） | a6b337d (4，只有文件) | §10.3 | `results/sanitize_590498/summary.md` |
| 511249 | test_grouped + memcheck | b56631d (0) | §2.2、§5.2、§5.3 | `results/slurm-ggemm-test-511249.out`、`test_511249/` |
| 511101 | numerics perf | 檔中未記錄 | §5.3 | `results/numerics_perf_511101.log` |
| 511065 | quick bench | da18425 (1) | §10.1、§10.4 | `results/bench_511065/bench.csv` |
| 511064 | nsys／ncu | da18425 (1) | §8.1、§8.2 | `results/prof_511064/analysis/` |
| 511063 | planner unit | 檔中未記錄 | §2.6、§5.2 | `results/planner_unit_511063/test_planner.log` |
| 511062 | sanitizer | da18425 (0) | §10.3 | `results/sanitize_511062/` |
| 511060 | graph／reuse／numerics | 檔中未記錄 | §7.4 | `results/slurm-ggemm-gr2-511060.out` |
| 511054、511001 | split-K | 見 `logs/*/provenance.txt` | §9.3 | `logs/i8cfg2sep_511054.log`、`logs/i8cfg2_511001.log` |
| 510965 | INT8 ablation | 5d8e2ad (18) | §9.2 | `logs/i8abl_510965.log` |
| 510850 | 校正 | — | §3 | `bench/calib/params_nvidia_h200.json` |
| 510881、510997、510961、511002、511003、511061 | 失敗的 job | 各自的 provenance | §10.6 | `results/slurm-*.out` |
