# HW3-1 報告（Naive DQN / Replay）

## 實驗設定摘要
- 環境：Gridworld 4x4
- 訓練任務：Naive DQN（static）、DQN+Replay（static）、DQN+Replay（random）
- 評估方式：以最後 10 回合平均指標作彙總

## 輸出位置
- 單一實驗輸出：`outputs/hw3-1/<exp_name>/`
- 匯總曲線：`outputs/summaries/loss_compare_hw3-1.png`
- 指標表：`outputs/summaries/metrics_table_hw3-1.csv`

## 指標彙總（最後 10 回合平均）
| Experiment | Mean Reward | Success Rate | Mean Steps | Mean Loss |
| --- | --- | --- | --- | --- |
| naive_static | -11.25 | 0.00 | 2.25 | 0.0056 |
| replay_static | 3.70 | 1.00 | 7.30 | 0.0016 |
| replay_random | -4.10 | 0.77 | 11.63 | 0.1143 |

## 曲線與動畫
### Loss 對照
![](../outputs/summaries/loss_compare_hw3-1.png)

### 策略動畫（Dashboard GIF）
**Naive DQN（static）**
![](../outputs/hw3-1/naive_static/dashboard.gif)

**DQN + Replay（static）**
![](../outputs/hw3-1/replay_static/dashboard.gif)

**DQN + Replay（random）**
![](../outputs/hw3-1/replay_random/dashboard.gif)

## 訓練期間策略優化（每 50 回合）
> 依各模型實際訓練總回合數，選取 $\approx$ 25% / 50% / 75% / 100% 的回合作為觀察點。

### Naive DQN（static，總回合 326）
| Episode 50 | Episode 150 | Episode 250 | Episode 300 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-1/naive_static/dashboards/episode_50.gif) | ![](../outputs/hw3-1/naive_static/dashboards/episode_150.gif) | ![](../outputs/hw3-1/naive_static/dashboards/episode_250.gif) | ![](../outputs/hw3-1/naive_static/dashboards/episode_300.gif) |

### DQN + Replay（static，總回合 552）
| Episode 150 | Episode 300 | Episode 400 | Episode 550 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-1/replay_static/dashboards/episode_150.gif) | ![](../outputs/hw3-1/replay_static/dashboards/episode_300.gif) | ![](../outputs/hw3-1/replay_static/dashboards/episode_400.gif) | ![](../outputs/hw3-1/replay_static/dashboards/episode_550.gif) |

### DQN + Replay（random，總回合 1000）
| Episode 250 | Episode 500 | Episode 750 | Episode 1000 |
| --- | --- | --- | --- |
| ![](../outputs/hw3-1/replay_random/dashboards/episode_250.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_500.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_750.gif) | ![](../outputs/hw3-1/replay_random/dashboards/episode_1000.gif) |

### 策略演進分析
- **Naive static**：早期仍偏向隨機探索，策略改進有限。
- **Replay static**：回合數提升後到達目標更穩定，路徑更一致。
- **Replay random**：策略逐漸避免陷阱，但仍受隨機環境干擾。

## 討論與比較
1. **Replay static** 明顯提升成功率並達到穩定收斂。
2. **Naive static** 收斂不穩定，成功率偏低。
3. **Replay random** 在隨機環境中仍能學到策略，但回饋較不穩定。

## 總結：為什麼需要 Replay Buffer
Replay Buffer 可打破樣本相關性，提升樣本效率與訓練穩定性，在 static 模式下特別明顯；在 random 模式下也能改善成功率。
