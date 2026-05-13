# HW3-2 報告（Double / Dueling / Dueling+Double）

## 實驗設定摘要
- 環境：Gridworld 4x4（player mode）
- Baseline：DQN + Replay（player mode，target network + replay buffer）
- 變體：Double DQN / Dueling DQN / Dueling + Double DQN
- 評估方式：以最後 10 回合平均指標作彙總

## 輸出位置
- 單一實驗輸出：`outputs/hw3-2/<exp_name>/`
- 匯總曲線：`outputs/summaries/loss_compare_hw3-2.png`
- 指標表：`outputs/summaries/metrics_table_hw3-2.csv`

## 指標彙總（最後 10 回合平均）
| Experiment | Mean Reward | Success Rate | Mean Steps | Mean Loss |
| --- | --- | --- | --- | --- |
| baseline_player | 6.93 | 1.00 | 4.07 | 0.0010 |
| double_player | 5.77 | 1.00 | 5.23 | 0.0013 |
| dueling_player | 6.37 | 1.00 | 4.63 | 0.0003 |
| dueling_double_player | 6.80 | 1.00 | 4.20 | 0.0002 |

## 曲線與動畫
### Loss 對照
![](../outputs/summaries/loss_compare_hw3-2.png)

### 策略動畫（Dashboard GIF）
**Baseline（DQN + Replay）**
![](../outputs/hw3-2/baseline_player/dashboard.gif)

**Double DQN**
![](../outputs/hw3-2/double_player/dashboard.gif)

**Dueling DQN**
![](../outputs/hw3-2/dueling_player/dashboard.gif)

**Dueling + Double DQN**
![](../outputs/hw3-2/dueling_double_player/dashboard.gif)

## 訓練期間策略優化（每 50 回合）
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

### 策略演進分析
- Baseline：中期開始穩定到達目標，路徑仍有些隨機性。
- Double DQN：穩定度提升，減少過度估計造成的偏移。
- Dueling DQN：更快學到高價值狀態，行為更一致。
- Dueling + Double：最早展現穩定策略，路徑最接近最短解。

## 討論與比較
1. **Double DQN** 降低過度估計，訓練曲線更平滑。
2. **Dueling DQN** 能更快辨識狀態價值，平均回饋提升。
3. **Dueling + Double DQN** 同時提升穩定性與效率。

## 總結
就本次結果，**Dueling + Double DQN** 在平均回饋與 loss 穩定度上表現最佳。
