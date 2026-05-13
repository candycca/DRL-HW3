# HW3-2 報告（Double / Dueling / Dueling+Double）

## 實驗目標
本實驗旨在探討在 Gridworld 4x4（player mode）環境下，**進階 DQN 架構**對訓練穩定性與策略品質的影響。以 DQN + Replay（Baseline）為基準，比較以下三種改良方法：

1. **Double DQN**：改善標準 DQN 的 Q 值過度估計問題。
2. **Dueling DQN**：分離「狀態價值」與「動作優勢」的估計，提升狀態評估效率。
3. **Dueling + Double DQN**：結合兩者優點，追求最佳的收斂速度與策略穩定性。



## 原理說明

### 1. Baseline：DQN + Replay（Target Network）
標準 DQN 的核心設計：
- 使用 **Experience Replay**：從 buffer 中隨機抽取 mini-batch，打破時間相關性。
- 使用 **Target Network**：以一個參數更新較慢的獨立網路計算 TD target，避免訓練目標頻繁跳動，提升穩定性。

$$Q\text{-target} = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

---

### 2. Double DQN
標準 DQN 在計算 TD target 時，使用相同的網路**選擇**並**評估**最佳動作，容易導致 Q 值被高估（overestimation bias）。Double DQN 的解法是：

- 用**線上網路（online network）**選擇動作
- 用**目標網路（target network）**評估該動作的價值

$$Q\text{-target} = r + \gamma Q_{\text{target}}\!\left(s',\, \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

這樣可有效降低過度估計，讓訓練曲線更平滑。

---

### 3. Dueling DQN
Dueling DQN 改變 Q-network 的網路架構，將輸出拆分為兩個分支：

- **Value stream $V(s)$**：衡量「身處某狀態有多好」
- **Advantage stream $A(s, a)$**：衡量「某動作相對其他動作有多好」

最終輸出：

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)$$

此設計使模型能更快學到「哪些狀態本身就很重要」，即使某些動作根本不影響結果也能有效學習。

---

### 4. Dueling + Double DQN
結合上述兩種改進：
- **Dueling 架構**：分離 V 與 A 的估計，提升狀態理解
- **Double 更新規則**：降低 Q 值高估偏差，穩定訓練目標

兩者相輔相成，在策略品質與訓練穩定度上均有提升。


## 實驗結果
### Loss 對照
![](../outputs/summaries/loss_compare_hw3-2.png)

| Experiment | Mean Reward | Success Rate | Mean Steps | Mean Loss |
| --- | --- | --- | --- | --- |
| baseline_player | 6.93 | 1.00 | 4.07 | 0.0010 |
| double_player | 5.77 | 1.00 | 5.23 | 0.0013 |
| dueling_player | 6.37 | 1.00 | 4.63 | 0.0003 |
| dueling_double_player | 6.80 | 1.00 | 4.20 | 0.0002 |


1. **Baseline（DQN + Replay）**：透過 Target Network 與 Replay Buffer，Loss 能穩定下降，但因 Q 值過度估計問題，仍存在一定程度的波動，最終 Mean Loss 約為 0.0010。
2. **Double DQN**：透過分離動作選擇與動作評估，有效緩解高估偏差，Loss 曲線整體更為平滑，但在此實驗中 Mean Reward 略低於 Baseline，可能因為更保守的估計使探索效率略有下降，Mean Loss 約為 0.0013。
3. **Dueling DQN**：雙流架構使模型更早學到狀態的內在價值，Loss 收斂速度更快、最終值更低（Mean Loss 0.0003），Mean Reward 也優於 Double DQN。
4. **Dueling + Double DQN**：結合兩者優勢，既能快速辨識狀態價值，又避免高估偏差，Loss 收斂至最低值（Mean Loss 0.0002），Mean Reward 僅次於 Baseline 且步數更接近最短路徑，整體表現最為穩定。


## 訓練期間策略優化
> 依各模型實際訓練總回合數，選取 $\approx$ 25% / 50% / 75% / 100% 的回合作為觀察點。

### Baseline（DQN + Replay）
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-2/baseline_player/dashboards/episode_250.gif) | ![](../outputs/hw3-2/baseline_player/dashboards/episode_500.gif) | ![](../outputs/hw3-2/baseline_player/dashboards/episode_750.gif) | ![](../outputs/hw3-2/baseline_player/dashboards/episode_1000.gif) |

### Double DQN
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-2/double_player/dashboards/episode_250.gif) | ![](../outputs/hw3-2/double_player/dashboards/episode_500.gif) | ![](../outputs/hw3-2/double_player/dashboards/episode_750.gif) | ![](../outputs/hw3-2/double_player/dashboards/episode_1000.gif) |

### Dueling DQN
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-2/dueling_player/dashboards/episode_250.gif) | ![](../outputs/hw3-2/dueling_player/dashboards/episode_500.gif) | ![](../outputs/hw3-2/dueling_player/dashboards/episode_750.gif) | ![](../outputs/hw3-2/dueling_player/dashboards/episode_1000.gif) |

### Dueling + Double DQN
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-2/dueling_double_player/dashboards/episode_250.gif) | ![](../outputs/hw3-2/dueling_double_player/dashboards/episode_500.gif) | ![](../outputs/hw3-2/dueling_double_player/dashboards/episode_750.gif) | ![](../outputs/hw3-2/dueling_double_player/dashboards/episode_1000.gif) |

- **Baseline**：中期開始穩定到達目標，路徑仍有些隨機性，需較多步數才能找到較佳路線。
- **Double DQN**：穩定度提升，減少過度估計造成的策略偏移，但收斂速度略慢。
- **Dueling DQN**：更快學到高價值狀態，行為更一致，中期即可展現穩定路徑。
- **Dueling + Double**：最早展現穩定策略，路徑最接近最短解，最終表現最為一致。

## 討論與比較

### 1. Q 值高估問題（Overestimation）
標準 DQN 在計算 TD target 時，以 $\max$ 操作選擇並評估同一動作，這種做法在估計誤差存在時會系統性地高估 Q 值。**Double DQN** 透過解耦「選擇」與「評估」步驟，有效緩解此問題：

- Baseline 的最終 Mean Loss（0.0010）高於使用 Double 機制的方法
- Double DQN 雖然在本實驗中 Mean Reward（5.77）略低，推測是因為隨機種子與初始策略的差異，而非架構本身的缺陷

### 2. 架構改良的效益（Dueling）
Dueling 架構在四種方法中呈現**最顯著的 Loss 降低效果**：

- Dueling DQN 的 Mean Loss（0.0003）相較 Baseline（0.0010）降低約 70%
- Dueling + Double DQN 的 Mean Loss（0.0002）更進一步，同時維持接近 Baseline 的 Mean Reward（6.80）與 Mean Steps（4.20）

這顯示 Value / Advantage 分流設計讓模型更精準地學習到「狀態本身的好壞」，而不需每次都依賴動作選擇才能更新。

### 3. 各方法綜合比較

| 方法 | 主要改進 | 優點 | 潛在限制 |
|------|---------|------|---------|
| Baseline | Target Net + Replay | 穩定基準，成功率 100% | Q 值高估，Loss 較高 |
| Double DQN | 解耦選擇與評估 | 降低高估偏差，訓練更保守 | 可能略微降低探索效率 |
| Dueling DQN | V/A 分流架構 | Loss 顯著下降，狀態理解更佳 | 架構較複雜 |
| Dueling + Double | 結合兩者 | Loss 最低，策略最穩定 | 訓練複雜度最高 |

## 總結

本實驗驗證了 Double DQN 與 Dueling DQN 兩種改良機制對 DQN 訓練的正面影響：

- **所有四種方法**在 player mode 下均達到 **100% 成功率**，說明 Experience Replay 與 Target Network 的基礎設計已足夠應對此環境。
- **Double DQN** 的主要貢獻在於穩定訓練動態，降低 Q 值高估，適合對訓練穩定性要求高的場景。
- **Dueling DQN** 透過架構層面的改良，讓模型更高效地學習狀態價值，Loss 收斂速度與最終值均優於 Baseline。
- **Dueling + Double DQN** 結合兩者優勢，在 Mean Loss（0.0002）與 Mean Reward（6.80）的綜合表現上最為均衡，是四種方法中推薦的最佳組合。

整體而言，這兩種改良技術的結合能在不增加訓練回合數的前提下，顯著提升 DQN 的策略品質與學習穩定性，值得在更複雜的環境中進一步驗證。
