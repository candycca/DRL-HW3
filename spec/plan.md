# HW3 規劃（Plan）

> 目標：依 `task/task.md` 完成 DQN 及其變體實作、訓練與報告輸出（曲線、GIF、量化指標、比較與總結）。

## 核心產出（全任務共用）
- 訓練 loss 曲線（多實驗對照）
- 訓練過程策略動畫（Dashboard GIF）
- 量化指標（例如：平均回饋、成功率、步數）
- 討論與比較

## 輸出資料夾結構（建議）
> 統一輸出位置，方便繪圖、比較與報告整理。

```
outputs/
	hw3-1/
		naive_static/
			metrics.csv
			loss.png
			dashboard.gif
			config.json
			checkpoints/
		replay_static/
			...
		replay_random/
			...
	hw3-2/
		baseline_player/
			...
		double_player/
			...
		dueling_player/
			...
		dueling_double_player/
			...
	hw3-3/
		lightning_baseline_random/
			...
		lightning_tips_random/
			...
	hw3-4/
		rainbow_random/
			...
	summaries/
		loss_compare_hw3-1.png
		loss_compare_hw3-2.png
		loss_compare_hw3-3.png
		loss_compare_hw3-4.png
		metrics_table_hw3-1.csv
		metrics_table_hw3-2.csv
		metrics_table_hw3-3.csv
		metrics_table_hw3-4.csv
```

### 建議輸出內容（每個實驗資料夾）
- `metrics.csv`：每回合指標（reward、loss、step、success）
- `loss.png`：單一實驗 loss 曲線
- `dashboard.gif`：訓練策略動畫
- `config.json`：超參數與環境設定
- `checkpoints/`：模型權重

## HW3-1：Naive DQN / Replay（Static + Random）
**目標**：完成三種設定並產出對照報告。

### 實作與執行
- Naive DQN（static mode）
- DQN + Replay（static mode）
- DQN + Replay（random mode）

### 產出
- 3 條 loss 曲線
- 3 個 Dashboard GIF
- 量化指標表
- 討論：基礎 DQN 在簡單環境的學習行為
- 比較：Replay Buffer 的效果
- 總結：為什麼需要 replay buffer

## HW3-2：Double / Dueling / Dueling+Double（Player）
**目標**：以 DQN + Replay（player）為 baseline，比較 3 種變體。

### 實作與執行
- Baseline：DQN + Replay（player mode）
- Double DQN（player mode）
- Dueling DQN（player mode）
- Dueling + Double DQN（player mode）

### 產出
- 4 條 loss 曲線
- 4 個 Dashboard GIF
- 量化指標表
- 討論：各變體改善點
- 比較：Baseline vs Double vs Dueling vs Dueling+Double
- 總結：最佳模型與原因

## HW3-3：Lightning + Training Tips（Random）
**目標**：把 DQN 轉成 PyTorch Lightning，並加入訓練技巧。

### 實作與執行
- Lightning 版 DQN + Replay（random）
- 加入訓練技巧（如：Gradient Clipping、LR Scheduler）

### 產出
- 訓練技巧原理簡述
- 消融 loss 曲線
- 消融 Dashboard GIF
- 量化指標
- 討論：Lightning 導入的設計改變
- 比較：Baseline vs 加入訓練技巧
- 總結

## HW3-4：Rainbow DQN（Random）
**目標**：以 Rainbow DQN 解 random mode，並與 baseline 對照。

### 實作與執行
- Rainbow DQN（random mode）

### 產出
- Rainbow DQN 介紹（技術與實作重點）
- loss 曲線：Baseline vs Rainbow
- Dashboard GIF：Baseline vs Rainbow
- 量化指標
- 討論：Lightning 轉換與 Rainbow 效果
- 比較與總結

## 建議的最小實驗規格（可依資源調整）
- 固定 random seed（至少 3 次平均）
- 訓練步數一致
- 儲存每 N 回合的統計與可視化
- 統一輸出目錄結構（例如：`outputs/<exp_name>/`）
 - **訓練需收斂**：至少觀察 reward/成功率趨勢穩定或 loss 下降並趨於平穩

## 腳本流程（建議）
> 依現有腳本分階段執行，統一由 `run_*.py` 產出結果。

1. **HW3-1（Static/Random）**
	- 使用 `src/run_hw3_1.py` 依序跑三種設定。
	- 產出：`outputs/hw3-1/*`。
2. **HW3-2（Player Mode 變體）**
	- 使用 `src/run_hw3_2.py` 跑 baseline + 3 變體。
	- 產出：`outputs/hw3-2/*`。
3. **HW3-3（Lightning + Training Tips）**
	- 使用 `src/run_hw3_4.py` 或 `src/run_ablation.py` 做消融（依實際腳本而定）。
	- 產出：`outputs/hw3-3/*`。
4. **HW3-4（Rainbow DQN）**
	- 使用 `src/run_hw3_4.py` 跑 Rainbow（若已與 HW3-3 分離，則新增專用腳本）。
	- 產出：`outputs/hw3-4/*`。

### 汇總與繪圖
- 以 `src/utils/metrics.py` 匯總 `metrics.csv`。
- 以 `src/utils/visualize.py` 繪製對照曲線與表格。
- 最終輸出到 `outputs/summaries/`。

## 驗收清單
- [ ] 四大作業皆可跑通
- [ ] 對應曲線、GIF、指標已產出
- [ ] 報告內容包含討論、比較、總結
