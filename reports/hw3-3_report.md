# HW3-3 報告（PyTorch Lightning DQN）

## 實驗目標
本實驗旨在將 HW3-2 的 DQN 訓練流程遷移至 **PyTorch Lightning** 框架，並比較加入工程最佳化技巧（Training Tips）前後的訓練效果。三個實驗均在 Gridworld 4x4（random mode）下執行，並以 HW3-1 的傳統 DQN + Replay（random）作為跨框架比較基準：

1. **BaseLine(HW3-1 DQN + Replay)**：HW3-1 的傳統 PyTorch 實作，作為跨框架基準。
2. **Lightning w/o Tips**：標準 Lightning DQN，使用固定學習率（Adam），無梯度裁剪，以三組 seed 匯總。
3. **Lightning w/ Tips**：加入 **CosineAnnealingLR** 學習率排程器與 **Gradient Clipping**（max norm = 1.0），探討工程調優對訓練穩定性的影響。

---

## PyTorch Lightning 框架說明

**PyTorch Lightning** 是建立在 PyTorch 之上的高階訓練框架，將訓練迴圈、優化器設定、日誌記錄等標準流程封裝為固定介面，讓研究者專注於模型邏輯本身。

在本實驗中，`LightningDQN` 繼承 `pl.LightningModule`，透過以下方式整合 RL 訓練：

| Lightning 介面 | 本實驗對應內容 |
|--------------|-------------|
| `configure_optimizers()` | 設定 Adam 優化器（與 CosineAnnealingLR） |
| `train_dataloader()` | 以 episode 索引為 batch，驅動訓練迴圈 |
| `training_step()` | 每回合執行環境互動、Replay Buffer 更新、Q-network 梯度更新 |
| `trainer.fit()` | 啟動完整訓練流程，自動管理 epoch 推進 |

---

## 兩種方法原理說明

### 1. Lightning w/o Tips
基礎實作，沿用標準 DQN + Replay Buffer + Target Network 架構，訓練超參數如下：

| 超參數 | 設定值 |
|-------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | `SmoothL1Loss`（Huber Loss）|
| γ (discount) | 0.95 |
| ε-greedy | 1.0 → 0.05（decay=0.995） |
| Replay capacity | 2000 |
| Batch size | 32 |
| Target update | 每 50 步 |
| Max episodes | 1000 |


### 2. Lightning w/ Tips    
在 Baseline 基礎上加入兩項工程最佳化技巧：

**① CosineAnnealingLR（學習率排程）**
學習率隨訓練進展從 `lr_max=0.001` 依餘弦曲線衰減至 `lr_min=0.0001`（= lr × 0.1），週期為 `T_max=1000`（等於 max_episodes），避免後期訓練時學習率過大導致 Loss 震盪：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T_{\max}}\right)$$

```python
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)
```

**② Gradient Clipping（梯度裁剪）**
在每次 backward 後，以 `clip_val=1.0` 限制梯度的 L2 norm，防止梯度爆炸（gradient explosion）在高方差樣本下破壞 Q-network 參數：

$$\hat{g} = g \cdot \frac{\min(1, \text{clip\_val} / \|g\|_2)}{1}
$$

```python
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
```
> **備注**：`SmoothL1Loss`（即 Huber Loss）在兩個實驗中均有使用，相較 MSE Loss 對離群值（outlier）更具強健性，是訓練 DQN 的常見良好實踐。


## 實驗結果

| Experiment | Mean Reward | Success Rate | Mean Steps | Mean Loss |
| --- | --- | --- | --- | --- |
| BaseLine(HW3-1 DQN + Replay) | -4.10 | 0.77 | 11.63 | 0.1143 |
| Lightning w/o Tips | 3.92 | 0.925 | 5.94 | 0.0810 |
| Lightning w/ Tips | 3.29 | 0.906 | 6.18 | 0.1190 |
### Loss 對照
![](../outputs/summaries/loss_compare_hw3-3_with_hw31.png)

**三種方法的訓練收斂情況說明：**
1. **Baseline（HW3-1 DQN+Replay）**：Loss 曲線整體偏高且波動較大（最終 Mean Loss 0.1143），受限於單次訓練與均勻梯度更新，在 random mode 高變異性環境下收斂較不穩定。
2. **Lightning w/o Tips**：三 seed 平均 Loss 曲線（0.0810）明顯低於 HW3-1 對照，收斂更為平滑。得益於 Early stopping 與 SmoothL1Loss，訓練在穩定後即自動停止，避免後期的 Loss 回升。
3. **Lightning w/ Tips**：加入 CosineAnnealingLR 後學習率遞減，後期 Loss 曲線更為平穩，但整體數值（0.1190）略高於 w/o Tips，顯示 Gradient Clipping 在此環境下限制了部分有效梯度，使收斂速度稍慢。

## 訓練期間策略優化
> 依各模型實際訓練總回合數（1000 回合），選取 $\approx$ 25% / 50% / 75% / 100% 的回合作為觀察點。
### BaseLine(HW3-1 DQN + Replay)（ random，總回合 1000）
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-1/replay_random/dashboards/episode_250.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_500.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_750.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_1000.gif) |

### Lightning w/o Tips（總回合 1000）
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-3/lightning_baseline_random/dashboards/episode_250.gif) | ![](../outputs/hw3-3/lightning_baseline_random/dashboards/episode_500.gif) | ![](../outputs/hw3-3/lightning_baseline_random/dashboards/episode_750.gif) | ![](../outputs/hw3-3/lightning_baseline_random/dashboards/episode_1000.gif) |

### Lightning w Tips（總回合 1000）
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-3/lightning_tips_random/dashboards/episode_250.gif) | ![](../outputs/hw3-3/lightning_tips_random/dashboards/episode_500.gif) | ![](../outputs/hw3-3/lightning_tips_random/dashboards/episode_750.gif) | ![](../outputs/hw3-3/lightning_tips_random/dashboards/episode_1000.gif) |

- **BaseLine(HW3-1 DQN + Replay)**：環境較複雜，策略不穩定。
- **Lightning w/o Tips**：從 Episode 250 起逐漸展現有效策略，Episode 750 後路徑趨於穩定，能在隨機環境中找到可行解。
- **Lightning w/ Tips**：早中期策略演進與 Baseline 相近，但因學習率隨訓練遞減，後期策略微調速度較慢；最終策略穩定性與 Baseline 相當。



## 討論與比較

### 1. PyTorch Lightning 框架的工程效益

Lightning 框架將訓練樣板（boilerplate code）封裝為標準介面，帶來以下工程優勢：

| 面向 | 傳統 PyTorch | PyTorch Lightning |
|------|------------|-------------------|
| 訓練迴圈 | 手動撰寫 for loop | `trainer.fit()` 自動管理 |
| 優化器 | 手動 zero_grad / backward / step | `automatic_optimization` 或手動模式均支援 |
| Multi-seed | 需手動包裝 | 可輕鬆迭代 seed，結果自動匯總 |
| 可重現性 | 需手動設定 seed | `set_seed()` 整合清晰 |

本實驗使用 `automatic_optimization=False`（手動模式），保留 RL 訓練的靈活性（如 gradient clipping 時機控制），同時享有 Lightning 的結構化優勢。

### 2. 與 BaseLine(HW3-1 DQN + Replay) 對照的跨框架比較

| 方法 | 框架 | Mean Reward | Success Rate | Mean Loss |
|------|------|------------|------------|----------|
| BaseLine(HW3-1 DQN + Replay) | 傳統 PyTorch | -4.10 | 0.77 | 0.1143 |
| Lightning w/o Tips | PyTorch Lightning | 3.92 | 0.925 | 0.0810 |
| Lightning w/ Tips | PyTorch Lightning | 3.29 | 0.906 | 0.1190 |

雖然 BaseLine(HW3-1 DQN + Replay) 與 HW3-3 的 Lightning w/o Tips 同樣使用 DQN + Replay，但兩者在 random mode 下的表現差距顯著。可能的原因包含：

- **Early stopping（收斂偵測）**：Lightning 版本設有收斂條件，訓練在穩定後停止，避免過擬合
- **損失函數**：Lightning 使用 `SmoothL1Loss`，對高 TD Error 樣本更具強健性

### 3. Training Tips 效益分析

| 技術 | 理論優勢 | 本實驗觀察 |
|------|---------|-----------|
| CosineAnnealingLR | 後期精細收斂，避免 loss 震盪 | Loss 整體略高，可能因 lr 衰減過快影響適應速度 |
| Gradient Clipping | 防止梯度爆炸，穩定大 batch 訓練 | 在此小型環境中效益不明顯，甚至略微限制學習效率 |

兩種 Tips 技術在**大規模、高維度**環境（如深度 CNN + Atari）中效益更為顯著。在 Gridworld 4x4 的小型環境中，梯度爆炸的風險本身較低，因此 Clipping 帶來的邊際效益有限。

從結果看，Baseline 在三組 seed 下均維持約 92% 的成功率，顯示其穩定性較高；Tips 版本的結果波動性略大（成功率 90.6%），但差距在可接受範圍內。



## 總結

本實驗從**跨框架比較**與**工程調優效益**兩個維度評估 PyTorch Lightning DQN 的表現：

- **HW3-1 vs Lightning 的跨框架比較**：在同一 random mode 環境下，Lightning  的成功率（92.5%）相較 HW3-1 傳統實作（77%）大幅提升，Mean Reward 也從 -4.10 上升至 3.92。此差距除框架本身外，也受益於多 seed 匯總、Early stopping 與 SmoothL1Loss 等設計。

- **Lightning w/o Tips  ** 以最簡配置達到 92.5% 成功率，證明 DQN + Replay 架構透過 Lightning 框架能有效訓練 random mode 環境，且基礎超參數已足夠勝任此任務。

- **Lightning w/ Tips** 在引入學習率排程與梯度裁剪後，成功率略降至 90.6%，提示這些技術在小型、低複雜度環境中的邊際效益有限，甚至可能因過度保守的梯度控制而輕微影響學習效率。

- 整體而言，Training Tips 技術**在更複雜的環境（高維度狀態空間、長時程任務）中才能充分發揮優勢**。未來若將此框架應用於更大型任務，建議保留 scheduler 但適當放寬 gradient clip 閾值（如從 1.0 提高至 5.0），以在穩定性與學習效率之間取得更佳平衡。
