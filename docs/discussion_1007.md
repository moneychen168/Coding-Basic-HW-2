# 10/7 第一次討論：Grouped GEMM on INT8 Tensor Core（Ozaki II）

> 說明：本大綱不含任何實測效能數字；所有尚未量測或查證的項目都標「待查證」或「待量測」。
> 評分對應：正確性與安全性 30%、效能與 profiling 30%、library 與可重現性 20%、結果分析與 scope 10%、討論與理解 10%。

---

## Slide 1：題目定義與動機

- Batched GEMM：B 個問題的 (m, n, k) 與 lda/ldb/ldc 都相同，一次 launch 處理。
- Grouped GEMM：每個問題有自己的 (m_g, n_g, k_g)、alpha_g、beta_g；G = 問題個數。常見場景：MoE expert、LoRA adapter、異質小矩陣。
- 目標：C_g = alpha_g * A_g * B_g + beta_g * C_g，輸出 FP64，column-major、不轉置（v1）。
- 為什麼選 RTX 5090（sm_120）：消費級 GPU 的 FP64 吞吐比 FP32 低很多（倍率待查證），INT8 Tensor Core 吞吐很高（TOPS 數字待查證）。因此用 INT8 模擬 FP64 有機會比 native FP64 快；但是否划算要看 shape，這正是 planner 的任務。
- 開發機是國網 H100/H200（sm_90），目標機 nano4 為 sm_120；同一份程式碼編兩個 arch。

**講稿：** 這題的核心是 Grouped 而不是單一 GEMM：每個問題尺寸都可能不同。我們選 RTX 5090 是因為它 FP64 弱而 INT8 強，所以用 INT8 模擬 FP64 才有意義；具體倍率我們還沒查證，所以投影片上不放數字。

---

## Slide 2：Ozaki scheme II（CRT 版）流程

符號定義：
- A (m x k)、B (k x n)：FP64 輸入。N：模數個數。p_1..p_N：兩兩互質、每個 ≤ 256 的整數模數。P = p_1 * p_2 * ... * p_N。
- A'、B'：A 每一列、B 每一行各乘一個 2 的冪次 scale 後截斷成的整數矩陣。條件：|A'B'| < P/2。
- A'_i = A' mod p_i、B'_i = B' mod p_i：以對稱範圍 [-p_i/2, p_i/2) 存成 int8 的 residue。
- C_i：INT8 GEMM、INT32 累加的結果，數學上等於 (A'B') mod p_i。

流程（4 步）：
1. Scale + 截斷：得到 A'、B'（位元數由 k 與 N 決定，確保不超過 P/2）。
2. 取 residue：對每個 p_i 算 A'_i、B'_i（int8）。
3. INT8 GEMM：C_i = A'_i * B'_i，INT32 精確累加（k ≤ 131072 才不溢位）。
4. CRT 重建：由 {C_i} 還原 A'B'（用 double-double 表示），再反 scale、套 alpha/beta，寫回 FP64 C。

**講稿：** 關鍵觀念是把一次 FP64 GEMM 換成 N 次 INT8 GEMM，每次都是精確整數運算，誤差只來自第一步的截斷。N 越大，P 越大，可容許的截斷位元越多，精度越高但成本越大。這也是為什麼 planner 要依 k 與目標誤差逐問題決定 N。

---

## Slide 3：Grouped 的難點

- Launch 數：EMU_LOOP 每個問題要 O(N) 次 kernel launch（N 個模數的 prep/GEMM/reconstruct），總共 O(G·N)；小問題下 launch 與 gap 可能超過計算時間（待量測）。
- Shape 異質：m、n、k 差很大，單一 tile 尺寸不可能對所有問題都好。
- Wave quantization：tile 總數與 SM 數不成整數倍時，最後一波 SM 閒置；問題之間工作量差異大時更明顯。
- Tile 選擇：大 tile 對大問題 reuse 好，但對小問題浪費；小 tile 則反之。

**講稿：** 這四點是 Grouped 比單一 GEMM 難的地方，也是我們設計的出發點。尤其 Ozaki 模擬會把每個問題再乘上 N 倍的 GEMM 數量，launch 問題比 native 更嚴重。

---

## Slide 4：我們的設計

- 3-kernel pipeline：prep（所有問題一次 launch，做 scale + residue）→ persistent grouped INT8 GEMM → reconstruct（CRT）。整組固定 3 個 kernel，共用一次 descriptor H2D。
- Persistent grouped INT8 GEMM：一個 kernel 跑完所有 (problem × modulus) 的 tile；tile scheduler 依工作量由大到小排序（LPT），改善 wave quantization；tile config 逐問題選擇（128x128 / 64x64 / 32x32）。
- Per-problem planner（AUTO）：每個問題各自決定 N；問題太小、不划算或 k 太大時 fallback 到 native FP64，並記錄在 stats。
- 安全性：計時區內不做 allocation，workspace 全部在 ggemm_create 配好；空間不足回傳錯誤碼而非偷偷 allocate。

**講稿：** 我們的主要貢獻是把 launch 數從 O(G·N) 降到常數 3，並用 LPT 排程處理異質 shape。Planner 讓小問題或不適合的問題走 native，避免模擬反而更慢或更不準。

---

## Slide 5：比較對象與公平性

- NATIVE_LOOP：每個問題呼叫一次 cublasDgemm（native FP64 baseline）。
- NATIVE_GROUPED：cublasDgemmGroupedBatched（native FP64 的 grouped baseline）。
- EMU_LOOP：starter-like，每個問題各呼叫單一 GEMM 的 INT8 模擬；用來隔離「launch 合併」的貢獻。
- Ours：EMU_GROUPED（固定 N）與 AUTO（planner）。
- 公平性：同一組輸入與同一 stream；計時區不含 allocation 與資料初始化；warmup 後取多次重複的統計（次數與統計量待定）；比較時同時看誤差，不只看時間。

**講稿：** 四條路徑用同一個 API 切換，所以輸入、stream、計時方式都一致。EMU_LOOP 對 Ours 能看出 launch 合併帶來多少，native 對 Ours 則回答模擬本身是否划算。

---

## Slide 6：測資設計與理由

- Uniform small：G 大、每個問題很小，主要壓測 launch overhead。
- MoE：同一 (n, k)，各 expert 的 m 不同（token 分配不均），壓測 shape 異質與 LPT。
- Random hetero：m、n、k 隨機，涵蓋 tile 選擇與 planner fallback 路徑。
- LoRA：低秩形狀（某一維很小，如 rank r），壓測不規則 tile 與 planner 判斷。
- 輸入分佈 phi：以 phi 控制元素動態範圍（例如 exp(phi · randn) 型態的縮放；具體定義見 bench 程式，待與 agent-B 對齊）。phi 大 → 動態範圍大 → 需要更多模數。

**講稿：** 四類測資各自對應一個設計假設。phi 這個參數很重要，因為 Ozaki 的誤差與所需 N 取決於動態範圍；如果只測 phi 小的資料會高估模擬的效益。

---

## Slide 7：正確性指標

- 參考解：高精度 reference（double-double 或 long double 的 CPU/GPU GEMM，待定），不用 native FP64 當 ground truth。
- 指標：逐元素最大相對誤差、Frobenius 範數相對誤差；與 native FP64 的誤差並列比較，目標是 FP64 等級（target_rel_err 預設 1e-15）。
- 邊界案例：k 不是 32 的倍數（pad 補 0）、beta = 0 與 beta ≠ 0、alpha = 0、m/n/k 為 1、極端動態範圍、含零矩陣。
- 安全性：無 out-of-bounds（compute-sanitizer，待跑）、workspace 不足時回傳錯誤、fallback 數量在 stats 可見。

**講稿：** 正確性占 30%，所以我們把它拆成數值誤差與記憶體安全兩塊。誤差一律對高精度 reference 評估，這樣才能公平比較 native 與模擬哪個更準。

---

## Slide 8：預計的 profiling

- nsys：看 timeline 上的 launch 數與 launch gap；預期 EMU_LOOP 有大量 gap、EMU_GROUPED 只有 3 個 kernel（這是待驗證的假設）。
- ncu：看 grouped INT8 GEMM 的 tensor pipe 使用率、memory throughput、achieved occupancy；分析 tile config 與 LPT 對 SM 閒置的影響。
- 分段計時：prep / GEMM / reconstruct 各占多少，找出下一個瓶頸（待量測）。
- 結果分析：用 roofline 觀點解釋哪些 shape 贏、哪些輸，並說明 planner 的 fallback 門檻如何決定。
- 所有 profiling 在 nano4 的 5090 上做；登入節點不執行任何 GPU 程式。

**講稿：** 效能與 profiling 占 30%，我們計畫用 nsys 證明 launch 問題、用 ncu 解釋 GEMM kernel 的效率。目前這些都是計畫，尚未取得數據。

---

## Slide 9：目前進度

- 狀態：實作中。
- 已完成：公開 API（include/ggemm.h）與內部設計契約（DESIGN.md）：資料佈局、模組分工、各 kernel 介面。
- 實作中：數值模組（scale/residue/CRT）、persistent grouped INT8 GEMM、pipeline 與 planner、bench/test 與 SLURM 腳本、related work 整理。
- 尚未完成：端到端正確性驗證、效能量測、profiling；因此本次不報告任何效能數字。

**講稿：** 目前還在實作階段，各模組依介面契約並行開發。我們選擇不放任何沒有實測的數字，效能結論留到下次。

---

## Slide 10：最後一週計畫與風險

- 計畫：
  1. 先在 H100/H200 用 SLURM 打通 correctness（sm_90 編譯）。
  2. 再到 nano4 的 5090 驗證 sm_120 並量測四條路徑。
  3. 跑 nsys/ncu，完成結果分析、planner fallback 門檻調整與 README/可重現腳本。
  4. 預留最後時間整理報告與簡報。
- 風險：
  - nano4 存取：排隊或權限問題，可能壓縮 sm_120 量測時間；緩解：sm_90 先完成所有 correctness，nano4 上只做量測。
  - Starter 整合：starter 介面與我們的 API 對接、EMU_LOOP 的公平重現可能花時間；緩解：先用自寫 starter-like 版本，再對接。
  - 禁止提交既有實作：GEMMul8 只讀不抄；程式碼需自行撰寫。
  - Scope：v1 不支援轉置；若時間不足，優先保證正確性與可重現性。

**講稿：** 最後一週的順序是先正確、再量測、再分析，因為正確性與可重現性占比高。最大的風險是 nano4 的存取與 starter 整合，所以我們把能在 H100 上完成的工作先做完。
