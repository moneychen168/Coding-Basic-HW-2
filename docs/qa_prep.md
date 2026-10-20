# 口試準備：核心程式、Profiling 解讀、參數變更預測

> 用途：口試時「解釋核心程式碼與 profiling 結果」以及「改一個參數或測資，先預測結果」。
> 數字一律引自 `results/` 與 `logs/` 底下的 job 目錄（標出 job id）。沒有量過的東西只寫推論，並標示【推論】。
> 程式行號以目前 working tree 為準（HEAD = b56631d）。

---

## 0. 符號與名詞（先定義，後文直接使用）

| 符號 / 名詞 | 定義 |
|---|---|
| G | 一次呼叫裡的問題個數（group size）。g = 0..G−1 是問題編號 |
| m, n, k | 問題 g 的形狀：C (m×n) = α·A (m×k)·B (k×n) + β·C，column-major，不轉置 |
| α, β | 每個問題自己的 alpha、beta |
| kp | `round_up(k, 32)`：INT8 GEMM 的 K 維度，尾端補 0 |
| N | 模數個數（num_moduli），2..20。N = 0 代表 adaptive（由程式自己選） |
| p_l | 第 l 個模數（l = 0..N−1），取 `OZ2_MODULI` 的前 N 個（256, 255, 253, 251, …），兩兩互質且 ≤ 256 |
| P | P = p_0·p_1·…·p_{N−1}，CRT 能表示的整數範圍 |
| T_N | 每一列 A′、每一行 B′ 的 2-norm 上限：T_N = sqrt(P/2·(1−2⁻²⁰))·(1−2⁻⁴⁰) |
| s_i, t_j | A 第 i 列、B 第 j 行的 2 的冪次 scale，取最大值使 2^{s_i}·‖A(i,:)‖₂ ≤ T_N |
| A′, B′ | 截斷後的整數矩陣：A′ = trunc(2^{s_i}·A)，B′ = trunc(2^{t_j}·B)。保證 \|A′B′\| < P/2 |
| plane | 一份 INT8 residue 矩陣。N 個模數的問題用 **N+1** 個 plane：l = 0..N−1 是 residue（A′ mod p_l，存成對稱 int8），l = N 是 abs plane |
| abs plane | A″ = floor(2^{σ_i}·\|A\|) ∈ [0,127]（σ_i 使該列最大值落在 [64,128)），B″ 同理。A″B″ 是 (\|A\|\|B\|) 的嚴格下界，給認證用 |
| C32 | INT8 GEMM 的 INT32 輸出，每個 (問題, plane) 一份 |
| E_ij | 每個輸出元素的嚴格誤差上界（reconstruct 內計算） |
| γ_n | FP64 dot product 的標準誤差常數 γ_n = n·u/(1−n·u)，u = 2⁻⁵³，n = k (+1 若 α≠1)(+1 若 β≠0) |
| 認證（certificate） | 檢查 \|α\|·E_ij ≤ γ_n·(\|α\|·L_ij + \|β·C0_ij\|)；L_ij 是 \|A\|\|B\| 的下界。通過 = 這個元素保證達到 native DGEMM 等級的 componentwise 誤差 |
| repair mode | OFF（不認證）/ COUNT（認證，失敗只計數 `num_uncertified`，保留模擬值）/ ON（失敗的元素在 kernel 內用 FP64 FMA dot 重算，計入 `num_repaired`） |
| safety fallback | A 列或 B 行含 NaN/Inf 的元素一律用 FP64 dot 計算（所有模式都做），計入 `num_safety_fallback` |
| φ (phi) | 測資動態範圍參數：a_ij = (U(0,1)−0.5)·exp(φ·N(0,1))。φ = 0 是均勻分布，φ = 2 約 17 個 binade |
| ρ (rho_budget) | 環境變數 `GGEMM_RHO_BUDGET`（預設 16），AUTO adaptive 選 N 時使用的 componentwise 預算 |
| tile_cfg | INT8 GEMM 的 tile：0 = 128×128，1 = 64×64，2 = 32×32，3 = 32×32 split-K（第二個 kernel） |
| LPT | Longest Processing Time first：把 tile 依每 tile 成本由大到小排，persistent CTA 依序領取 |
| slots | persistent grid 同時能跑的 CTA 數 = SM 數 × 每 SM CTA 數（H200：132×2 = 264） |
| launches | `stats.kernel_launches`：一次呼叫發出的 kernel 數 |
| fb | `stats.num_fallback`：因 K 太大 / 精度達不到 / tiny / 不划算而改走 native FP64 的問題數（empty、k=0、α=0 不算） |
| path 0..6 | 0 NATIVE_LOOP、1 NATIVE_GROUPED、2 EMU_LOOP、3 EMU_GROUPED、4 AUTO、5 EMU_GROUPED_REUSE、6 EMU_LOOP_BATCHED（README 表格） |

結果 job 對照（provenance.txt）：

| job | 內容 | git |
|---|---|---|
| 590147 | 最終 bench（全部 path，φ=0.5，adaptive N，reps 20）、nsys/ncu、test_* | b56631d（= HEAD） |
| 590148 | repair sweep：EMU_GROUPED × N × {COUNT, ON} × φ；AUTO × ρ | b56631d |
| 511065 | 精度 sweep：φ × 固定 N（reps 3） | da18425（早於 prep split-K 修改） |
| 511064 | nsys/ncu 摘要（`analysis/*.md`） | da18425 |
| 510965 / 511054 | INT8 kernel ablation / split-K 決策 | 5d8e2ad+dirty / — |
| 511101 | numerics 誤差界表（`results/numerics_perf_511101.log`） | b56631d 前後 |
| 511063 | planner 單元測試（含 host 規劃時間） | da18425 |

注意：590147 的 nsys/ncu 只有原始 trace（`final_590147/analysis/` 是空的），階段百分比摘要只有 511064 那一份。

---

## 1. 核心程式導讀（約一小時讀完）

### 1.1 呼叫流程總覽

```
ggemm_grouped_dgemm (ggemm.cu:695)
  ├─ ggemm_validate (ggemm.cu:83)          參數、workspace 上限檢查
  ├─ repair mode 決定 (ggemm.cu:56-65)     AUTO→ON，其他 EMU→COUNT
  ├─ NATIVE_*  → path_native (ggemm.cu:344)
  └─ EMU_* / AUTO → path_emu (ggemm.cu:373)
        1. ggemm_plan (planner.cpp:228)          每個問題：route / N / tile_cfg
        2. [path 5] build_operand_reuse          找共用 A/B
        3. 切分 + workspace offset (ggemm.cu:404-465)
        4. 填 descriptor 到 pinned slot (ggemm.cu:482-570)
        5. LPT schedule image (ggemm.cu:574-596)
        6. 一次 H2D (ggemm.cu:601-606)
        7. launch：prep → i8 GEMM → reconstruct (ggemm.cu:660-682)
        8. native 問題：cublasDgemmGroupedBatched (ggemm.cu:685-689)
```

### 1.2 Handle 與 workspace（`ggemm_create_ex`，ggemm.cu:228-339）

- 所有 device / pinned 記憶體都在這裡配好，計時區不 allocate。
- A8 / B8 / C32 依「最大 plane 數 21 × 元素上限 × kp 膨脹係數」配置（ggemm.cu:266-281）；kp 膨脹在 k < 32 時最嚴重（k=1 時 32 倍），所以有 `min_k` 參數。
- descriptor blob 是 **雙 slot 的 pinned buffer**（`desc_done[2]`，ggemm.cu:39-54、296-300）：第 s 個 slot 要等上上次用它的呼叫在 stream 上完成（`cudaEventSynchronize`）才能覆寫。這是 host 端在連續呼叫時偶爾會被阻塞的原因。
- `oz2_init_constants()`（ggemm.cu:253）把 CRT 常數表上傳到 `__constant__`，避免計時區內有 H2D。
- `i8_grouped_gemm_reserve`（ggemm.cu:287）預先做 `cudaFuncSetAttribute` 與 occupancy 查詢。

**資料佈局**（DESIGN.md「內部資料佈局」、ggemm.cu:404-540）：
- `A8[plane][prob]`：每個問題 m×kp row-major（K 連續）。
- `B8[plane][prob]`：每個問題 n×kp row-major（等於 col-major 的 kp×n）。
- `C32[plane][prob]`：每個問題 m×n col-major（ld = m）。
- plane 之間的 stride（SA、SB、SC）是「所有問題該 plane 大小的總和」，所以一個 plane 是一整塊 slab，第 l 個 plane 的位址 = plane 0 位址 + l·stride。
- `row_shift[4m]` = {s_i | σ_i | ‖ΔA_i‖ 上界 (float bits) | ‖A′_i‖ 上界 (float bits)}，`col_shift[4n]` 同理（`OZ2_ROW_WS_PER_ROW` = 4）。

### 1.3 Planner / AUTO（planner.cpp、planner.h）

- `ggemm_plan`（planner.cpp:228-277）對每個問題依序判斷 route（planner.h 第 6-16 行的 enum）：
  1. m=0 或 n=0 → EMPTY（不做事）
  2. k=0 → K_ZERO（cuBLAS 算 β·C）
  3. α=0 → ALPHA_ZERO（cuBLAS 算 β·C，所有 EMU path 都這樣）
  4. k > 131040 → K_TOO_LARGE（allow_fallback=0 時整個呼叫回 `GGEMM_ERR_UNSAFE`）
  5. 只有 AUTO：k ≤ 16 → SMALL_K（planner.cpp:245，planner.h 的 `GGEMM_AUTO_NATIVE_MAX_K`）
  6. 決定 N：AUTO adaptive 用 `oz2_choose_num_moduli_cw(k, ρ)`（componentwise 規劃，ozaki2_numerics.cu:644-652）；其他 path 用 normwise 的 `oz2_choose_num_moduli(k, 1e-15)`（ozaki2_numerics.cu:631-636）。normwise 選法在 k = 256..4096 選 N=15，k = 16384、65536 選 16，k = 131072 回 −1（job 511101 表）。
  7. 只有 AUTO：`tile_by_padding`（planner.cpp:153-162，padding 效率 ≥ 0.75 的最大 tile）、`cost`（planner.cpp:175-201）、m·n·k < 32³ → TINY、t_emu > t_native → UNPROFITABLE。
- `tune_tiles`（planner.cpp:206-226）：若某問題單一 tile 的工作量大於「總工作量 / slots」，就把它的 tile 縮小一級（避免單一大 tile 拖尾），再重算一次 cost。
- **成本模型**（planner.cpp:175-201），符號：np = N+1，bw = 記憶體頻寬，i8 = INT8 吞吐，f64 = FP64 吞吐：
  - prep = [2·(mk+kn)·8 + np·(m+n)·kp + 16·(m+n)] / bw
  - gemm = ceil(np·tiles/slots)·(2·t²·kp)/(i8·eff[tile]/slots) + np·m·n·4/bw，eff = {1.0, 0.63, 0.30}
  - recon = (np·m·n·4 + c_rw·m·n·8)/bw，c_rw = 2（β≠0）或 1
  - native = max(2mnk/f64, (mk+kn+c_rw·mn)·8/bw)
  - 參數來源：`bench/calib/params_nvidia_h200.json`（f64 = 6.13e13、i8 = 1.676e15、bw = 3.94e12，job 510850 量測），否則用 planner.cpp:98-104 的 nominal 值。
- `GGEMM_RHO_BUDGET` 是 function-static（planner.cpp:232），整個 process 只讀一次。

### 1.4 prep kernel：scale + residue（ozaki2_numerics.cu:242-440，launcher 588-602）

- **一次 launch 處理整組**：grid = (a_blocks + b_blocks, num, S)。a_blocks = ceil(max_m/32)，b_blocks = ceil(max_n/8)，**用整組的最大 m、n**，小問題多出來的 block 一進來就 return（ozaki2_numerics.cu:267、368）。
- A 部分（263-362）：一個 block 處理 32 列。先算列最大值與平方和（必要時用 2 的冪次 prescale 避免 overflow，292-299），決定 s_i（`shift_from_norm`，145-152）與 σ_i；再以 32×32 的 k tile 轉置，對每個元素做 `prep_elem`（trunc、累加 ‖ΔA‖²、‖A′‖²）+ N 個 `residue_sym`，寫進 smem（XOR swizzle）後以 4 byte 一次寫出所有 N+1 個 plane（337-342）。
- B 部分（363-439）：一個 warp 處理一行（B 本來就是 K 連續），4 個元素 pack 成一個 uint32 寫出。
- `residue_sym`（193-201）：把整數 A′ 拆成 2^E·(c0 + c1·2¹⁸ + c2·2³⁶)，查表 2^e mod p，Barrett 約簡，轉成對稱 int8。**每個元素、每個模數都要做一次**，這是 prep 的主要整數運算量。
- prep split-K（595-599）：grid 塊數 < 4×SM 時把 k 範圍切成 S 份（最多 16），每份重算列範數，結果 bit-exact（job 511101 的 commit 加入）。

### 1.5 Persistent grouped INT8 GEMM（int8_grouped_gemm.cu）

- 介面：`i8_problem {m, n, kp, A, B, C, tile_cfg}`，C = A·Bᵀ（A m×kp、B n×kp 都 K 連續），INT32 精確累加。限制 kp ≤ 131040（128²·kp < 2³¹）。
- **一個 kernel 跑完所有 (問題 × plane) 的 tile**（`i8_grouped_persistent_kernel`，360-407）：grid = slots（H200 = 264，ncu raw 確認 grid (264,1,1)）；每個 CTA 用 atomicAdd 從全域計數器領 tile id（預取下一個），在 prefix-sum 表上二分搜尋找到所屬 entry（378-386），再依 tile_cfg 呼叫 `run_tile<Cfg>`。最後一個 CTA 把計數器歸零（398-406），所以不需要 memset launch。
- 三種 tile 共用 256 threads、48 KB smem、3 stage（77-92）：cfg0 128×128 BK=64、cfg1 64×64 BK=128、cfg2 32×32 BK=256。
- `run_tile`（160-257）：cp.async 3-stage pipeline → ldmatrix（XOR swizzle）→ `mma.sync.m16n8k32.s8.s8.s32` → col-major epilogue。超出 kp 的 k32 step 整段跳過（204）。
- LPT 排程在 host 建（`build_schedule_impl`，602-675）：每 entry 成本 = bm²·(kp+128)（`tile_cost`，492-495），穩定排序由重到輕；split-K（cfg3）的 entry 排在後面、另成一張表、另一個 kernel（413-450）。
- **注意**：`i8_choose_tile_cfg`（517-530）、group-level fill rule（624-639）、auto split-K（614-619）都只在 `tile_cfg < 0` 時生效。ggemm 的 planner 一定給 0..2 的明確值（ggemm.cu:532），所以從 ggemm API 呼叫時這三者都不會發生。EMU_GROUPED / REUSE 一律 tile 0（planner.cpp:258），只有 AUTO 會換 tile。

### 1.6 reconstruct：CRT + 認證 + repair（ozaki2_numerics.cu:452-565，launcher 604-616）

- grid = (gx, num)，gx = min(1024, ceil(max_m·max_n/1024))，每 thread 約 4 個元素，grid-stride 走訪（同樣用整組最大 m·n）。
- 每個元素：
  1. α=0 → β·C0；任一側 NaN/Inf → safety fallback FP64 dot（493-499）；任一側全零 → β·C0（501-504）。
  2. CRT（505-520）：y_l = (C32_l mod p_l)·q_l mod p_l；S1 = Σ s1_l·y_l（uint64）、S2 = Σ s2_l·y_l（uint32）；Q = rint((S1 + S2·2⁻¹⁸)·u1/P)；X/u1 = (S1 − Q·P1) + (S2·2⁻¹⁸ − Q·P2)。這是「int64 + double」的 double-double 等級運算。
  3. v = X·2^{−(s_i+t_j)}（2 的冪次，exact），emu = α·v + β·C0（523）。
  4. OFF 模式到此結束（525-530）。
  5. 認證（531-545）：E = 2^{−(s+t)}·[ΔA·‖B′‖ + ‖A′‖·ΔB + ΔA·ΔB + CRT 誤差 + 4u\|X\|]；L = abs plane 的 INT32 值 × 2^{−(σ+τ)}，再與 \|v\|−E 取大；檢查 \|α\|E ≤ γ_n(\|α\|L + \|βC0\|)。
  6. 不通過：COUNT 只 ++n_uncert；ON 呼叫 `fp64_dot`（443-450，單一 thread 逐 k 做 FMA，A 以 lda 跨步讀取）。
- 計數器：**每個 thread 只要計數非零就對同一個全域位址做一次 atomicAdd**（562-564），不論 `GGEMM_STATS_REPAIR` 是否開啟（那個變數只控制要不要讀回）。

### 1.7 Operand reuse（path 5；operand_reuse.h/.cpp，ggemm.cu:389-399、426-445、541-566）

- 在 host 端以雜湊表找相同的 A-key = (A 指標, lda, m, k[, N]) 與 B-key = (B 指標, ldb, n, k[, N])（operand_reuse.cpp:47-126，O(G)）。
- 為什麼 key 要含 N：scale s_i 由 T_N 決定，用 N_max 算的 residue 對 N 較小的問題不合法（operand_reuse.h:12-24）。預設 EXACT_N；`GGEMM_REUSE_PROMOTE=1` 改成把相連的問題都升到同一個最大 N（union-find）。
- prep 改成跑「unique A 的 A-only descriptor（n=0）＋ unique B 的 B-only descriptor（m=0）」（ggemm.cu:541-566），所以 prep 的 grid y 從 G 變成 UA+UB。GEMM 與 reconstruct 不變。
- 結果與 EMU_GROUPED bit-exact（test_reuse_e2e「bit-exact REUSE vs EMU_GROUPED: yes」）。

### 1.8 CUDA Graph plan / execute（ggemm_graph.cu）

- `ggemm_make_plan_ex`（259-301）→ `build`（143-257）：做一次 planner、配置 plan 自己的 device/pinned blob、填 descriptor、建 LPT、一次同步 H2D，然後在私有 stream 上 capture `enqueue`（75-104：prep、i8、recon、native），instantiate 並 `cudaGraphUpload`。
- `ggemm_execute_plan`（309-333）：只有 `cudaGraphLaunch`；若用別的 stream，前後各用一個 event 與 handle stream 串接（共用 A8/B8/C32 scratch 與自重置的 LPT 計數器）。
- `ggemm_plan_update_pointers`（335-386）：使用者的 A/B/C 指標不是 kernel 參數（kernel 只拿 blob 位址），所以換指標只要 patch host blob + 一次小 H2D；唯一例外是只有 1 個 native 問題時（cuBLAS 參數），需要重新 capture + `cudaGraphExecUpdate`。
- 限制：只支援 path 3、4；不做 operand reuse；形狀 / α / β 固定在 plan 裡。

---

## 2. Profiling 結果怎麼解讀

### 2.1 端到端時間（job 590147，H200，φ=0.5，adaptive N，median of 20，單位 ms）

| workload | native_loop | native_grouped | emu_loop | emu_loop_batched | emu_grouped | emu_grouped_reuse | auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni_G1024_s32 | 7.0810 | 0.0683 | 117.0723 | 36.4513 | 1.2510 | 1.3990 | 0.1845 |
| moe_E64_prefill_z1.2_up | 1.6484 | 1.5760 | 43.4219 | 22.4916 | 6.8033 | 6.7089 | 1.5943 |
| het_G256_16-2048 | 3.0616 | 1.9742 | 59.4523 | 23.9898 | 10.2095 | 11.3317 | 2.0185 |
| lora_G64_rmix_shrink | 0.8325 | 0.1660 | 54.4046 | 17.3831 | 1.7343 | 1.7327 | 0.1722 |

launches（同一 job）：uni_G1024_s32 的 emu_loop = 18432、emu_loop_batched = 3072、emu_grouped = 3、auto = 1；moe_E64 的 emu_loop = 1152、batched = 192。AUTO 在 **所有** bench workload 上 `fb = G`（全部改走 native）。

解讀：
1. **合併 launch 是成立的**：emu_grouped 相對 emu_loop 快 5.8–94×（het 5.8×、MoE 6.4×、lora 31×、uni_G1024_s32 94×；問題越小、越多，倍數越大），相對 GEMMul8 式的 emu_loop_batched 快 2.3–29×。
2. **但在 H200 上模擬本身不划算**：emu_grouped 比 native_grouped 慢（MoE 4.3×、het 5.2×、uni_G1024_s32 18×）。原因是 H200 的 FP64 很強（calibration 6.13e13 FLOP/s），而 Ozaki-II 要做 N+1 = 16 個 INT8 plane 加上 prep/recon 的記憶體搬運。
3. **AUTO 做了正確決定**：全部 fallback 到 native，時間 ≈ native_grouped（例如 MoE 1.5943 vs 1.5760）。

### 2.2 時間花在哪：各階段（nsys）

**job 511064 摘要**（`prof_511064/analysis/nsys_summary_*.md`，git da18425，**在 prep split-K 修改之前**）。stage % = 佔 GPU busy 時間的比例：

| workload / path | kernels/call | GPU span (µs) | launch gaps (µs) | host plan (µs) | prep % | INT8 GEMM % | recon % |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni_G1024_s32 emu_loop | 18432 | 113,365 | 17,776 | 5794.7 | 12.6 | 71.5 | 15.8 |
| uni_G1024_s32 emu_grouped | 3 | 577.5 | 6.0 | 675.7 | 30.4 | 52.8 | 10.1 |
| moe_E64_up emu_grouped | 3 | 6671.1 | 5.3 | 38.7 | 63.4 | 29.6 | 6.9 |
| het_G256 emu_grouped | 3 | 9640.2 | 5.5 | 213.5 | 39.1 | 36.9 | 23.9 |
| lora_G64_shrink emu_grouped | 3 | 2046.6 | 5.4 | 38.7 | 86.5 | 12.7 | 0.5 |
| lora_G64_shrink emu_loop_batched | 192 | 56,818 | 262.9 | 64.4 | 94.6 | 4.4 | 1.0 |
| uni_G1024_s32 native_loop | 1024 | 10,720 | 8646.9 | 0 | – | – | – |

**job 590147 原始 trace**（`final_590147/prof/*_cuda_gpu_trace.csv`，HEAD 版本），emu_grouped 單次呼叫的三個 kernel：

| workload | prep | INT8 GEMM | reconstruct | 加總 | bench 端到端 |
|---|---:|---:|---:|---:|---:|
| uni_G1024_s32 | 0.203 ms（grid 5×1024） | 0.299 ms（grid 264） | 0.054 ms | 0.557 ms | 1.251 ms |
| moe_E64_prefill_z1.2_up | 4.31–4.34 ms（206×64） | 1.97–2.03 ms | 0.476–0.492 ms | ≈ 6.8 ms | 6.803 ms |
| het_G256_16-2048 | 4.09–4.17 ms（316×256） | 3.54–3.70 ms | 2.32–2.40 ms | ≈ 10.0 ms | 10.210 ms |

解讀：
- **MoE / LoRA：prep 是第一瓶頸**（MoE 4.3 / 6.8 ms ≈ 63%）。MoE 的 B（64 個 expert × 1408 × 2048）主導：prep 讀 FP64 兩遍、寫 16 個 INT8 plane。
- **het：三段都重**，reconstruct 佔 23%：grid 用整組 max_m×max_n，加上 46.6M 個輸出元素都要做 N 次 Barrett + 認證。
- **小問題、大 G（uni_G1024_s32）：host 端是瓶頸**。GPU kernel 只有 0.557 ms，端到端 1.251 ms；511064 的 host plan = 675.7 µs（規劃 + 16384 個 i8 descriptor + LPT 排序 + 填 blob）。planner 單元測試（job 511063）量到規劃本身約 110–123 ns/問題。
- emu_loop 的問題是 launch：18432 個 kernel、17.8 ms 的 gap、83.8 ms 的 host API 時間（511064）。

### 2.3 kernel 內部（ncu，job 511064 摘要，median）

| workload | kernel | 時間 (µs) | INT8 tensor pipe % | DRAM % | SM % | 前三大 stall |
|---|---|---:|---:|---:|---:|---|
| uni_G1024_s32 | i8 GEMM | 370.6 | **4.9** | 4.5 | 39.5 | long_scoreboard 35%、wait 33% |
| uni_G1024_s32 | prep | 228.7 | – | 2.5 | 66.6 | short_scoreboard 31%、mio_throttle 20% |
| moe_E64_up | i8 GEMM | 2450.3 | 44.5 | 31.6 | 48.8 | wait 29%、short_scoreboard 18% |
| moe_E64_up | prep | 5521.0 | – | 23.6 | **70.0** | not_selected 29%、math_pipe_throttle 23% |
| het_G256 | i8 GEMM | 4195.1 | 23.3 | 23.8 | 35.4 | wait 24%、long_scoreboard 23% |
| het_G256 | reconstruct | 2744.1 | – | 25.2 | 59.8 | long_scoreboard 46% |
| lora_G64_shrink | prep | 2055.0 | – | 5.3 | 18.5 | wait 36% |

（ncu 在 replay 下鎖頻量測，時間不能直接和 bench 比。i8 kernel 的 achieved occupancy ≈ 25%，即 2 CTA × 8 warps / 64 warps；128 regs/thread。）

解讀：
- **prep 是整數 ALU bound，不是頻寬 bound**：MoE prep 的 SM throughput 70%、math_pipe_throttle 23%，DRAM 只有 23.6%。每個元素要做 N 次（3 個乘法 + Barrett + 對稱化），N=15 時約上百個整數運算 / 元素。
- **INT8 GEMM 用不滿 tensor core**：最好 44.5%，tiny 形狀只有 4.9%。原因：EMU_GROUPED 固定 128×128 tile，32×32 的問題只用到 1/16 的 tile；kp=32 時只有一個 k step，時間花在 prologue / epilogue 與 latency（long_scoreboard）。
- **kernel 本身的天花板**（job 510965，i8 kernel 單獨量）：single 8192³ 我們 612 TOPS、cuBLASLt 1466 TOPS（約 42%）；grouped random64 我們 cfg0 441 TOPS、cuBLASLt 逐一呼叫 110 TOPS（4×）；small256 我們 0.0259 ms、cuBLASLt loop 2.267 ms。原因：用 `mma.sync`（Ampere 式），沒有用 sm_90 的 wgmma / TMA。grouped 的優勢來自不必逐問題 launch。
- **reconstruct 是 latency bound**（long_scoreboard 46–54%）：每個元素讀 N+1 個 INT32（stride = 整個 plane slab）。

### 2.4 一句話結論（口試版）

在 H200 上，Grouped 化把 launch 數從 O(G·N) 壓到 3、比逐問題模擬快 6–94 倍（小問題大 G 時最多）；但 Ozaki-II 的 prep（整數 ALU）與 N+1 倍的 INT8 工作量，讓模擬比 H200 原生 FP64 慢 4–18 倍，所以 AUTO 的成本模型把所有問題都送回 native。模擬路徑內部，大問題的瓶頸是 prep，小問題大 G 的瓶頸是 host 規劃，tensor core 使用率最高只有 45%。

---

## 3. 參數 / 測資變更預測表

標記：**【實測】** = 有 job 數據直接支撐；**【推論】** = 只根據成本模型或程式邏輯推出，沒有量過。

**驗證方式的共同說明**：
- `bench_grouped` **沒有 `--help`**，不認得的參數會印 `unknown arg` 並 exit 2。可用參數：`--set bench|quick|test|repair|tiny`、`--paths`、`--phis`、`--moduli`、`--reps`、`--warmup`、`--csv`、`--only`、`--no-check`、`--moe-counts file:n:k`、`--dump-shapes`、`--dump-only`、`--repair-modes default,off,count,on`、`--count-call`、`--plan-csv`（bench/bench_grouped.cu:136-151）。
- bench 固定 α=1、β=0（bench_grouped.cu:215），EMU 系列 `allow_fallback=0`、AUTO `allow_fallback=1`（:226）。
- 下面的指令都要包在 SLURM job 裡跑（例如複製 `scripts/run_bench.slurm` 的 header，或 `sbatch -A GOV114009 -p dev --gpus-per-node=1 --cpus-per-task=12 --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn085 --wrap '…'`），不可在 login node 執行。`B=build/bench_grouped`。
- 自訂形狀：`--moe-counts file:n:k` 會讀一個每行一個整數的檔案，產生 m_e = 該整數、(n, k) 固定的問題（bench/workloads.cu:135-146）。這是做 m=1、空問題、k 很大等測試最簡單的方法。

### Q1. 模數個數 N 從 14 改成 18

- **時間**：EMU_GROUPED 增加約 **+18% 到 +23%**。【實測，job 590148，COUNT 模式】uni_G1024_s64 φ=0.5：1.5283 → 1.8852 ms（+23%）；moe_E64_up φ=0.5：6.4215 → 7.7946 ms（+21%）；het_G256 φ=0.5：9.7453 → 11.4697 ms（+18%）；uni_G256_s256 φ=0.5：2.3761 → 2.8652 ms（+21%）。
- **為什麼不是 +27%**：plane 數 15 → 19（+27%），INT8 工作量與 residue / C32 流量都正比於 N+1；但 prep 讀 FP64 的部分、host 規劃與 launch 不變，所以總增幅小於 27%。
- **launches**：EMU_GROUPED 不變（3）。EMU_LOOP 每個問題 N+3 次（prep + N+1 個 plane + recon），所以 +4G。【推論，公式在 ggemm.cu:610-636；N=14 時 4352 = 256×17 已由 590147 驗證】
- **誤差**：φ ≤ 1 時已經在 ~1e-16 的底部，幾乎不變（uni_G1024_s64 φ=0.5：5.97e-16 → 1.46e-16）；φ=2 時明顯改善：COUNT 模式 1.72e-12 → 1.31e-15，uncertified 2,389,316 → 1170。【實測 590148】
- **驗證**：`$B --set repair --only uni_G1024_s64 --paths emu_grouped --phis 0.5,2 --moduli 14,18 --repair-modes count --count-call --reps 10 --csv q1.csv`，或 `MODULI=14,18 PHIS=0.5,2 sbatch scripts/repair_sweep.slurm`。

### Q2. N 小於 14（例如 8、10、12）

- **誤差**：每少 2 個模數，誤差約大 2–3 個數量級。【實測 511065，uni_G1024_s32 φ=0.5】N=8 6.13e-9、N=10 3.41e-11、N=12 1.42e-13、N=14 5.95e-16。與誤差界表一致（511101，k=1024：N=8 5.0e-8、N=12 1.0e-12、N=14 5.4e-15、N=15 4.9e-16）。
- **時間**：plane 少了應該變快，uni_G1024_s32 確實如此（N=8 0.889 ms、N=14 1.081 ms）。但 MoE / het / lora_expand 反而**變慢**（het：N=12 23.75 ms、N=14 9.22 ms）。【實測 511065】
- **為什麼會變慢（推論，與數據吻合，沒用 ncu 直接驗證）**：N 小時大部分元素認證失敗，reconstruct 的每個 thread 都對同一個位址做 atomicAdd（ozaki2_numerics.cu:562-564），這些 atomic 會被序列化。估算：uni_G1024_s64 有約 1.05M 個 thread 各做一次，590148 量到 COUNT N=13（4,193,592 個 uncertified）比 N=14（0 個）多 0.586 ms，約每個 atomic 0.56 ns ≈ 1 個 clock，符合「同位址 atomic 序列化」。
- **驗證**：用 `--repair-modes off,count` 比較同一個 N（OFF 不做 atomic）；或用 ncu 看 `lts__t_requests_op_atom` / L2 atomic throughput。`$B --set repair --only het_G256 --paths emu_grouped --phis 0.5 --moduli 12,14 --repair-modes off,count --reps 10 --csv q2.csv`。

### Q3. k 加倍（m、n 不變）

- **時間**：INT8 GEMM 工作量與 prep 的 FP64 讀取、residue 寫入都加倍；reconstruct（只跟 m·n、N 有關）不變。所以 GEMM + prep 部分約 ×2，總時間約 ×1.6–×2（reconstruct 佔比越大，增幅越小；het 的 recon 佔 23%，增幅會偏低）。【推論；bench 沒有「只改 k」的成對 workload，uniform 的 s→2s 是 m、n、k 一起變】
- **N 是否改變**：adaptive normwise 在 k = 256..4096 都選 15，k ≥ 16384 選 16（511101），所以通常不變；k 從 16 變 32 會從 N=14 變 15（590147：uni_G256_s16 launches 4352 = 256×17 → N=14；s32 4608 = 256×18 → N=15）。【實測】
- **誤差**：γ_n ∝ k，認證門檻變寬；emu 誤差界 ≈ 4√k/T + 4k/T²，在 N=15 時仍遠小於 γ_k，uncertified 預期減少或不變。native 的誤差隨 k 增加，所以 vs_native 比值變小。【推論】
- **非 32 倍數的 k**：kp = round_up(k,32)，k=33 → kp=64，INT8 工作量幾乎 ×2（test set 的 uni_G64_7x13x33 正確性已過，590147 test_grouped）。
- **驗證**：做兩個 counts 檔（例如每行 256，共 64 行），`$B --moe-counts c.txt:1024:1024 …` 與 `--moe-counts c.txt:1024:2048 …`，`--paths emu_grouped,native_grouped`。

### Q4. k > 131040（例如 k = 140000）

- **EMU_GROUPED / EMU_LOOP / REUSE（bench 裡 allow_fallback=0）**：planner 回 `GGEMM_ERR_UNSAFE`（rc = 3），bench CSV 的 status = `call_failed`，不會算出結果。【實測，590147 test_grouped：「bigk emu_grouped allow_fallback=0 -> refused (rc=3)」】
- **AUTO（allow_fallback=1）**：該問題 route = K_TOO_LARGE，走 native；`num_fallback` +1；launches：若 native 問題 ≥ 2 個就 1 次 grouped cuBLAS，否則每個 1 次。【實測：test_grouped「bigk AUTO fallback=3 err/gamma=1.885e-01」】
- **邊界**：判斷的是 k，不是 kp：k = 131040 仍模擬（kp = 131040，128²·131040 < 2³¹），k = 131041 就 fallback（planner.cpp:241、internal.h `GGEMM_MAX_K_EXACT`）。
- **驗證**：counts 檔每行 4，`$B --moe-counts c.txt:64:140000 --paths emu_grouped,auto --reps 3 --csv q4.csv`。

### Q5. G 從 16 增加到 1024，每個問題很小（16³）

- **launches**：native_loop = G、emu_loop = G·(N+3)、emu_loop_batched = 3G、emu_grouped = 3（不隨 G 變）、AUTO = 1。【實測 590147：uni_G256_s16 emu_loop 4352、batched 768；uni_G1024_s16 17408、3072；G4096 69632、12288】
- **時間**：emu_grouped **約與 G 成正比**，不是常數：G = 256 / 1024 / 4096 → 0.2654 / 1.0381 / 4.6938 ms。原因：host 規劃與 descriptor 是 O(G)、GPU 工作也是 O(G)，而且每個 16×16 問題都占一個 128×128 tile、16 個 plane。emu_grouped 對 emu_loop 的加速穩定在 86–95×（24.14/0.2654、98.97/1.0381、403.48/4.6938）。【實測 590147】
- **AUTO**：全部走 native（k=16 ≤ 16 先命中 SMALL_K；16³ = 4096 < 32768 也會命中 TINY），fb = G，時間 ≈ native_grouped（G1024：0.0686 vs 0.0643 ms）。【實測】
- **G 的上限**：emulated 問題 > 65535 時回 `GGEMM_ERR_INVALID`（ggemm.cu:463，prep / recon 的 grid y 上限）。【程式邏輯】
- **驗證**：`$B --set bench --only uni_G --paths emu_loop,emu_grouped,native_grouped,auto --reps 20 --csv q5.csv`（--only 是子字串比對）。

### Q6. 動態範圍變大：φ 從 0.5 變成 2

- **時間**：native 不變；EMU_GROUPED 在 COUNT 且 uncertified 少的時候幾乎不變（het N=16：10.455 → 10.416 ms），uncertified 多時因 atomic 變慢（het N=14：9.745 → 19.857 ms）；ON 模式會因 repair 大幅變慢（moe N=15：COUNT 6.7876 vs ON 22.4619 ms）。【實測 590148】
- **誤差**：adaptive N（=15）在 φ = 0 / 0.5 / 1 / 2 時，uni_G1024_s32 的誤差為 1.69e-16 / 1.89e-16 / 3.06e-16 / 5.71e-14（native 是 5.00e-16 / 6.83e-16 / 9.32e-16 / 1.15e-15）。也就是 φ=2 時 COUNT 模式下模擬誤差變成 native 的約 50 倍。【實測 511065】
- **為什麼**：planner 只看 (k, target) 選 N，不看資料動態範圍（planner.cpp:130-136 的註解：要看 span 就得在計時區做 D2H + sync）。動態範圍大時 ρ_ij = ‖a_i‖‖b_j‖/(|A||B|)_ij 變大，normwise 界不能保證 componentwise 精度，需要靠認證 + repair。
- **COUNT 模式不保證精度**：test_grouped（590147）uni_G256_s16 φ=2 的 emu_grouped err/γ = 162.7、uncertified = 39090，測試仍然通過，因為 COUNT 的規則是「回報」不是「修正」。要達到 native 等級必須用 ON：590148 的 φ=2 ON 模式誤差都在 native 的 0.2–1.5 倍。
- **驗證**：`$B --set quick --only uni_G1024_s32 --paths native_loop,emu_grouped --phis 0,0.5,1,2 --moduli 0 --repair-modes count,on --count-call --reps 5 --csv q6.csv`。

### Q7. 改 α、β（α=1.5、β=−0.75，或 α=0）

- **α≠1、β≠0**：reconstruct 多讀一次 C0（每個元素 +8 B），門檻 n 多 2；其他三個 stage 不變。時間只在 reconstruct 增加，總時間預期增加 < 5%（recon 只佔 GPU 時間 7–24%，見 2.2）。【推論】誤差幾乎不變。【實測 590147 test_grouped，uni_G32_s256 φ=1：a1_b0 err_cw 2.038e-16 → a1.5_b−0.75 2.062e-16；het_G32 φ=1：uncertified 3954 → 2543】
- **β=0**：不讀 C（C 是 NaN 也不影響，edge/beta0_nanC 通過）。
- **α=0**：route ALPHA_ZERO，該問題不做 prep / GEMM，改由 cuBLAS 算 β·C；如果整組都是 α=0，EMU_GROUPED 的 launches 變 1（只有 cuBLAS），且不算 fallback。【程式邏輯：planner.cpp:240、ggemm_route_is_fallback】
- **驗證**：bench 不能改 α/β；用 `build/test_grouped --set test --phis 1`（φ=1 時自動多跑 α=1.5、β=−0.75，tests/test_grouped.cu:120-121），看 `[…/a1.5_b-0.75]` 區塊。

### Q8. 有空問題（m=0 或 n=0）、k=0

- **m=0 / n=0**：route EMPTY，不產生任何 tile、不 launch；全部為空時 launches = 0、時間 ≈ host 開銷。【實測 590147 test_grouped edge/all_empty：所有 path launches=0】
- **k=0**：route K_ZERO，交給 cuBLAS 算 β·C（exact），**launches +1**（EMU_GROUPED 變 4），`num_fallback` 不加。【實測：edge/zero_dims_mix emu_grouped launches=4、fallback=0；test_graph mixed16(+empty,+k0) EMU_GROUPED 4 launches】
- MoE decode 有空 expert 時 native_loop 的 launches 會少：moe_E128_decode_z0.0 是 126（128 個 expert 中 2 個 m=0）。【實測 590147】
- **驗證**：counts 檔混 0 與正數，`$B --moe-counts c.txt:256:512 --paths emu_grouped,auto,native_loop --reps 5`；k=0 要用 test_grouped 的 edge case。

### Q9. m = 1（decode 式的 GEMV）

- **INT8 GEMM 浪費最大**：每個問題 1 列卻用 128×128 tile（EMU_GROUPED），mma 最小 M 也是 16，至少 16 倍浪費；工作量主要在 prep 的 B（n×k）。【推論】
- **AUTO 一定 fallback（H200 與 5090 都是）**：native 是讀 B 一次的 memory bound（≈ 8·n·k / bw），而 prep 光是讀 B 兩次再寫 N+1 個 plane（≈ 16·n·k + 16·n·k bytes）就超過，所以 t_emu > t_native → UNPROFITABLE（planner.cpp:187-200）。若 m·n·k < 32768 會先命中 TINY。【推論，由成本模型】
- 正確性：edge/one_1x1x1 已測（590147 test_grouped，所有 path ok）。
- **驗證**：counts 檔 64 行都是 1，`$B --moe-counts ones.txt:1408:2048 --paths native_grouped,emu_grouped,auto --plan-csv q9_plan.csv --reps 10`，看 plan CSV 的 route 欄（5 = UNPROFITABLE、4 = TINY）。

### Q10. 均勻形狀 vs 異質形狀

- **異質時 prep / reconstruct 的效率較差**：兩個 kernel 的 grid 都用整組 max_m、max_n（ozaki2_numerics.cu:592-600、610-614），小問題的 block 大多直接 return；EMU_GROUPED 又固定 128 tile。INT8 GEMM 有 LPT，影響比較小。【程式邏輯】
- 【實測】het_G256 的 reconstruct 佔 GPU busy 23.9%，MoE 只有 6.9%（511064）；590147 trace：het recon 2.32–2.40 ms、MoE 0.48 ms。het 的 i8 tensor pipe 23.3%，MoE 44.5%（511064 ncu）。
- **預測**：把同樣總 FLOPs 的 uniform 換成異質，emu_grouped 變慢，而且 recon/prep 的比例上升；AUTO 的 per-problem tile（tile_by_padding + tune_tiles）在 emu 被選中時能減少浪費，但在 H200 上 AUTO 全部 fallback，看不到。【推論】
- **驗證**：`$B --set bench --only het_G --paths emu_grouped,native_grouped --reps 20`；搭配 `sbatch scripts/profile.slurm`（`WORKLOADS="het_G256 uni_G256_s256"`）看各 kernel 時間。

### Q11. Repair OFF / COUNT / ON

- **uncertified = 0 時三者幾乎一樣**：COUNT 與 ON 差 < 1%（het N=16 φ=0.5：10.4551 vs 10.4297 ms）。【實測 590148】OFF 少做認證的浮點運算，預期再快一點，但 **沒有 OFF 的計時數據**（590148 只跑 count,on）。【推論】
- **uncertified 很多時**：ON 要做 FP64 dot（每個元素一個 thread 跑 k 次 FMA，A 以 lda 跨步讀），非常慢：moe_E64 φ=2 N=14 COUNT 8.6558 → ON 34.3754 ms（4,271,165 個 repair）；het φ=2 N=14 19.8566 → 46.8874 ms。【實測 590148】
- **誤差**：COUNT 的誤差可遠大於 native（het φ=2 N=13：3.28e-11，native 的 4.9e3 倍）；ON 修回 native 等級（9.14e-15，1.4 倍）。OFF 與 COUNT 的輸出 bitwise 相同。【實測 590148；test_numerics「OFF==COUNT bitwise:1」，slurm-ggemm-gr2-511060.out】
- **預設**：EMU 系列 = COUNT，AUTO = ON（ggemm.cu:56-65）。
- **驗證**：`$B --set repair --paths emu_grouped --phis 2 --moduli 14,16 --repair-modes off,count,on --count-call --reps 10 --csv q11.csv`。

### Q12. Operand reuse 開 / 關（path 5 vs path 3）

- **沒有共用 operand 時 reuse 比較慢**：bench 的每個問題都有自己的 A、B 緩衝區（bench/workloads.cu:296-305），找不到可共用的；prep 改成 A-only + B-only 兩組 descriptor，grid y 從 G 變 2G。uni_G1024_s32：1.2510 → 1.3990 ms（+12%）；trace 中 prep 0.203 → ~0.30 ms（grid 5×1024 → 5×2048）；het_G256：10.2095 → 11.3317 ms（+11%）。【實測 590147】
- **有共用時**：省下的時間上限 = prep 佔比 × 被共用 operand 在 prep 位元組中的比例。【實測 590147 test_reuse_e2e】lora（G=16，B 共用，uB=1）1563.3 → 1498.3 µs（1.043×）；tp（G=8，A 共用，uA=1）1515.5 → 1380.1 µs（1.098×）；none（無共用）596.9 → 557.2 µs（1.071×，這個數字和 bench 的「變慢」方向相反，原因沒有查證）。
- **GGEMM_REUSE_PROMOTE=1**：不同 N 的問題也能共用，但會把 N 升到相連分量的最大值，INT8 工作量可能增加（stats.num_promoted）。【推論】
- **驗證**：`build/test_reuse_e2e`（tests/test_reuse_e2e.cu）；或 `$B --set quick --paths emu_grouped,emu_grouped_reuse --reps 20`。

### Q13. 改 tile config（例如強制 32×32 或 64×64）

- **大形狀**：tile 越小越慢。cfg1 約為 cfg0 的 0.6×、cfg2 約 0.27–0.29×（single 8192³：583.7 / 360.0 / 164.8 TOPS；random64：440.8 / 290.4 / 128.9 TOPS）。【實測 510965】
- **小形狀（m,n ≤ 200）**：cfg0 與 cfg1 差不多（0.0259 vs 0.0257 ms），cfg2 反而慢（0.0455 ms）。【實測 510965】
- **對 uni_G1024_s32 的 EMU_GROUPED**：目前 128 tile 讓 tensor pipe 只有 4.9%。換成 32×32 tile 會減少無效 MMA，但 GEMM 只占 0.299 / 1.251 ms，而且 host 規劃約 0.68 ms 不變，所以端到端最多改善約 20%。【推論】
- **怎麼改**：EMU_GROUPED 在程式裡寫死 tile 0（planner.cpp:258），沒有參數可調；AUTO 才會選 tile，但 H200 上 AUTO 全部 fallback。可以用 `GGEMM_FP64_FLOPS=1e11 GGEMM_TINY_MNK=0`（讓成本模型以為 FP64 很弱）強迫 AUTO 模擬，tile 就由 `tile_by_padding` 決定。或直接用 i8 kernel 的 ablation：`sbatch scripts/i8_ablation.slurm`（cfg0/1/2 強制版本）。

### Q14. 開 INT8 split-K（tile config 3）

- **透過 ggemm API：沒有任何效果**。`i8_set_auto_splitk_min_kp` 只影響 `tile_cfg < 0`（自動選 tile）的問題（int8_grouped_gemm.cu:614-619），而 ggemm 的 planner 一律給 0..2 的明確值（ggemm.cu:532）。launches 不變。【程式邏輯】
- **直接呼叫 i8 kernel 時**：auto-sk 讓 small256 慢 31%（0.0236 → 0.0310 ms）、mix 慢 3%；launches 1 → 2。單獨比較時 cfg3 比 cfg2 快（small256 0.0432 vs 0.0455 ms），但在 auto 裡 32×32 tile 只是少數，搬到第二個序列化的 kernel 反而失去填尾巴的效果。【實測 511054，int8_grouped_gemm.cu:27-39 的紀錄】
- 不要和 **prep split-K** 混淆：prep 在 grid 塊數 < 4×SM 時自動切 k（ozaki2_numerics.cu:595-599），結果 bit-exact。
- **驗證**：`sbatch scripts/i8_cfg2sep.slurm`（auto vs auto-sk）。

### Q15. 改用 CUDA Graph（plan/execute）而不是直接呼叫

- **省的是 host 端**：host issue 時間從 85.84 µs 降到 1.96 µs（tiny64x(32,32,64)，EMU_GROUPED）；wall 86.49 → 68.55 µs（−21%）。大問題幾乎沒差：moe32 611.18 → 596.78 µs（−2%）。AUTO mixed16：107.70 → 73.75 µs。launches 不變。【實測 590147 test_graph】
- **預測 uni_G1024_s32**：端到端 1.251 ms 中約 0.68 ms 是 host 規劃，用 graph 預期接近 GPU kernel 時間 0.56 ms 加上 graph launch 開銷，約快 2 倍。【推論】
- 限制：只支援 path 3、4；graph 不做 operand reuse；形狀固定。
- **驗證**：`build/test_graph`（tests/test_graph.cu），印出 direct / graph 的 host_issue_us、wall_us。

### Q16. 改 GGEMM_RHO_BUDGET（4 / 16 / 64 / 256）

- **在 H200 上完全沒有影響**：AUTO 把所有問題 route 成 UNPROFITABLE（5）或 TINY（4），根本沒有用到 ρ；四個 ρ 的時間與誤差都相同（例如 het_G256 φ=2：2.0112 / 2.0044 / 2.0124 / 2.0096 ms，err 8.06e-15）。【實測 590148 auto_rho*.log、auto_plan.csv】
- 若強迫 AUTO 模擬（GGEMM_FP64_FLOPS 調低）：ρ 大 → N 大 → repair 少、plane 多；ρ 小 → 反之。正確性不受影響（repair 會補）。【推論】
- 注意 ρ 是 process 內只讀一次（planner.cpp:232），所以每個 ρ 要一個獨立的 bench process（repair_sweep.slurm 的 Part B 就是這樣跑）。

### Q17. 換到 RTX 5090（sm_120）

**全部是推論**：nano4 是 H200，沒有任何 5090 量測；`bench/calib/` 只有 `params_nvidia_h200.json`，5090 上 planner 會用 planner.cpp:100-101 的 nominal 值（FP64 1.64e12×0.8、INT8 838e12×0.5、頻寬 1.79e12×0.8；其中 INT8 838 TOPS 是否為 dense 在程式註解中標為未驗證）。

- **native FP64**：比 H200 calibration 值（6.13e13）低約 37 倍，compute-bound 的形狀 native 會慢很多。
- **emu**：INT8 nominal 約 H200 的一半，頻寬 0.45×，所以 prep / recon（記憶體與整數 ALU）預期比 H200 慢約 2 倍，GEMM 也慢約 2 倍。
- **AUTO 的決定會翻轉**：以 MoE E64 up 的一個 expert（m≈96, n=1408, k=2048, N=15）代入 planner.cpp:175-201：H200 上 t_nat ≈ 9 µs，光 prep 就約 25 µs → UNPROFITABLE；5090 nominal 下 t_nat ≈ 420 µs，t_emu ≈ 70（prep）+ 60（GEMM）+ 7（recon）≈ 140 µs → EMU。所以預測 `num_fallback` 從 G 大幅下降，AUTO 的 launches 從 1 變成 3（或 4，若還有 native 問題）。
- **仍會 fallback 的**：k ≤ 16（SMALL_K）、m·n·k < 32768（TINY）、m=1 這類 memory-bound 形狀（見 Q9）。
- **風險**：sm_120 有編譯但從沒跑過；i8 kernel 宣稱 48 KB smem × 2 CTA/SM 也放得下（int8_grouped_gemm.cu:45），未驗證；cost model 的 `GGEMM_I8_CTAS_PER_SM` 固定 2。
- **驗證**：5090 上先跑 `bench/calib`（`sbatch bench/calib/run_calib.slurm` 產生 params_*.json），再跑 `sbatch scripts/run_bench.slurm` 與 `--plan-csv` 看 route 分布。也可以在 H200 上用 `GGEMM_FP64_FLOPS=1.31e12 GGEMM_INT8_OPS=4.19e14 GGEMM_MEM_BW=1.43e12` 模擬「planner 以為是 5090」，只驗證 planner 決策（kernel 速度仍是 H200 的）。

### Q18（補充）. 只把 β 從 0 改成非 0 時，AUTO 的決定會變嗎？

- 成本模型的 c_rw 從 1 變 2，native 與 emu 都多算 8·m·n bytes；兩邊增加的量一樣，所以只有在兩者非常接近時才會翻轉。H200 上差距很大，預期 fb 不變。【推論】

---

## 4. 程式行為與文件不一致之處（口試時可能被問到）

1. **README「INT8 split-K … enable it with `i8_set_auto_splitk_min_kp`. When enabled, EMU_GROUPED can issue 4 launches」**：從 ggemm API 做不到。auto split-K 只對 `tile_cfg < 0` 的問題生效（int8_grouped_gemm.cu:614-619），ggemm 的 planner 一律給明確的 0..2（planner.cpp:236、259、222；ggemm.cu:532）。EMU_GROUPED 出現 4 個 launch 的真正原因是有 k=0 / α=0 / fallback 問題走 cuBLAS（test_grouped edge/zero_dims_mix）。
2. **README Known limitations「Plan/execute does not merge same-shape problems in its native fallback」**：已過時。graph 的 build 會呼叫 `ggemm_group_native`（ggemm_graph.cu:234-238），同形狀會合併成一個 cuBLAS group。
3. **README 環境變數表「GGEMM_RHO_BUDGET：planner budget for the cost-based decisions」以及表頭「Read at ggemm_create」**：實際上是 `ggemm_plan` 第一次被呼叫時以 function-static 讀一次、整個 process 固定（planner.cpp:232），而且只用在 AUTO adaptive 選 N（`oz2_choose_num_moduli_cw`），和 cost-based fallback 無關。
4. **README「GGEMM_CALIB_JSON … (`f64_peak_flops`, `i8_peak_ops`, `mem_bw_*` scalars)」**：程式讀的頻寬 key 是 `bw_Bps`（planner.cpp:89）。
5. **README path 表把「LPT tile scheduling」寫在 AUTO**：LPT 在所有 grouped path（3、4、5）都有（int8 `build_schedule_impl`），不是 AUTO 獨有。README 也沒寫 AUTO 的 SMALL_K route（k ≤ 16 直接 native，planner.cpp:245）。
6. **計數器的成本**：README 讓人以為 `GGEMM_STATS_REPAIR` 沒開就沒有成本；但 reconstruct 在 COUNT/ON 模式每個 thread 都會對同一個位址 atomicAdd（ozaki2_numerics.cu:562-564），不論是否讀回。590148 中 COUNT 模式在 uncertified 很多時明顯變慢（Q2）。
7. **DESIGN.md「tile config 依問題大小逐個選」**：只有 AUTO 如此；EMU_GROUPED / REUSE 一律 128×128（planner.cpp:258）。int8 kernel 裡的 `i8_choose_tile_cfg` 與「tile 數 < SM 數時改 64×64」的 fill rule（int8_grouped_gemm.cu:624-639）從 ggemm 呼叫時永遠不會觸發。
8. **docs/discussion_1007.md**：寫「k ≤ 131072 才不溢位」、目標機器 RTX 5090；程式是 k ≤ 131040（`OZ2_MAX_K`），實際平台是 H200（README 已更正，discussion 檔是 10/07 的舊稿）。
9. **成本模型的 INT8 吞吐偏樂觀**：calibration 的 `i8_peak_ops` = 1.676e15，但我們的 kernel 實測最多約 0.61e15（510965 single 8192³）。在 H200 上不影響結論（光 prep 就輸），換到 FP64 較弱的 GPU 時，planner 可能高估模擬的優勢。
10. 小事：ggemm.cu 第 1 行註解寫「the 5 execution paths」，實際有 7 條。
