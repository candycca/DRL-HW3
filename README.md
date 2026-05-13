# RL HW3 – DQN and Variants

本專案包含 HW3-1 的基礎 DQN/Replay 實作與輸出流程，並保留後續 HW3-2~HW3-4 的擴充空間。

## HW3-1 輸出結構
執行 `src/run_hw3_1.py` 會在 `outputs/hw3-1/` 產生：

```
outputs/hw3-1/
  naive_static/
    seed_0/
    seed_1/
    seed_2/
    metrics.csv
    loss.png
    dashboard.gif
    config.json
    checkpoints/
  replay_static/
    ...
  replay_random/
    ...
outputs/summaries/
  loss_compare_hw3-1.png
  metrics_table_hw3-1.csv
```

## 內容包含
- `metrics.csv`：episode-level 指標（reward、loss、steps、success）
- `loss.png`：loss 曲線
- `dashboard.gif`：策略動畫（以訓練後的 greedy policy 生成）
- `config.json`：超參數與環境設定

## 後續作業
請參考 `spec/plan.md` 與 `spec/hw3-*.md` 的規劃與輸出格式。

## HW3-2 輸出結構
執行 `src/run_hw3_2.py` 會在 `outputs/hw3-2/` 產生：

```
outputs/hw3-2/
  baseline_player/
  double_player/
  dueling_player/
  dueling_double_player/
outputs/summaries/
  loss_compare_hw3-2.png
  metrics_table_hw3-2.csv
```

## HW3-3 輸出結構
執行 `src/run_hw3_3.py` 會在 `outputs/hw3-3/` 產生：

```
outputs/hw3-3/
  lightning_baseline_random/
  lightning_tips_random/
outputs/summaries/
  loss_compare_hw3-3.png
  metrics_table_hw3-3.csv
```
