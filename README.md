# RL HW3 – DQN 與進階變體實作

本專案實作並比較多種深度 Q 網路（DQN）演算法，包含 Naive DQN、Experience Replay、Double DQN、Dueling DQN、PyTorch Lightning 框架整合、以及 Rainbow DQN，實驗環境為 Gridworld 4×4。

---

## 目錄結構

```
RL_HW3/
├── src/
│   ├── Gridworld.py              # 環境定義
│   ├── GridBoard.py              # 棋盤邏輯
│   ├── dqn_common.py             # 共用元件（QNetwork、TrainConfig、ε-greedy 等）
│   ├── dqn_naive.py              # Naive DQN（無 Replay Buffer）
│   ├── dqn_replay.py             # DQN + Experience Replay + Target Network
│   ├── double_dqn.py             # Double DQN
│   ├── dueling_dqn.py            # Dueling DQN
│   ├── dueling_double_dqn.py     # Dueling + Double DQN
│   ├── lightning_dqn.py          # PyTorch Lightning DQN
│   ├── rainbow_dqn.py            # Rainbow DQN（Double + Dueling + NoisyNet + PER + n-step）
│   ├── run_hw3_1.py              # HW3-1 執行腳本
│   ├── run_hw3_2.py              # HW3-2 執行腳本
│   ├── run_hw3_3.py              # HW3-3 執行腳本
│   └── run_hw3_4.py              # HW3-4 執行腳本
├── outputs/
│   ├── hw3-1/                    # HW3-1 各實驗輸出
│   ├── hw3-2/                    # HW3-2 各實驗輸出
│   ├── hw3-3/                    # HW3-3 各實驗輸出
│   ├── hw3-4/                    # HW3-4 各實驗輸出
│   └── summaries/                # 跨實驗匯總圖表與 CSV
├── reports/
│   ├── hw3-1_report.md           # HW3-1 實驗報告
│   ├── hw3-2_report.md           # HW3-2 實驗報告
│   ├── hw3-3_report.md           # HW3-3 實驗報告
│   └── hw3-4_report.md           # HW3-4 實驗報告
├── checkpoints/                  # 模型 checkpoint 儲存
├── spec/                         # 作業規格文件
├── requirements.txt
└── README.md
```

---

## 實驗環境（Gridworld 4×4）

- **狀態空間**：Player、Goal、Pit、Wall 的位置組合
- **行為空間**：上 / 下 / 左 / 右（4 個離散動作）
- **獎勵設計**：Goal = +10、Pit = -10、其他步驟 = -1
- **終止條件**：抵達 Goal（成功）或掉入 Pit（失敗）

**Mode 說明**：

| Mode | 說明 |
|------|------|
| `static` | Goal、Pit、Wall 與 Player 初始位置固定 |
| `player` | Goal、Pit、Wall 固定，Player 初始位置隨機 |
| `random` | 所有元素（Goal、Pit、Wall、Player）隨機初始化 |

## 實驗概覽
### HW3-1：Naive DQN vs Experience Replay
**執行腳本**：`src/run_hw3_1.py`  
**環境**：Gridworld 4×4（static / random mode）  
**比較方法**：

| 方法 | Mode | 說明 |
|------|------|------|
| Naive DQN | static | 無 Replay Buffer，online 逐步更新 |
| DQN + Replay | static | 加入 Experience Replay + Target Network |
| DQN + Replay | random | 場景隨機初始化，測試泛化能力 |

**報告**：[reports/hw3-1_report.md](reports/hw3-1_report.md)

---

### HW3-2：Double / Dueling / Dueling+Double DQN
**執行腳本**：`src/run_hw3_2.py`  
**環境**：Gridworld 4×4（player mode）  
**比較方法**：

| 方法 | 說明 |
|------|------|
| Baseline (DQN + Replay) | 含 Target Network，作為基準 |
| Double DQN | 解耦動作選擇與評估，降低 Q 值高估 |
| Dueling DQN | V/A 雙流架構，提升狀態理解效率 |
| Dueling + Double DQN | 結合兩者，綜合效益最佳 |

**報告**：[reports/hw3-2_report.md](reports/hw3-2_report.md)

---

### HW3-3：PyTorch Lightning DQN
**執行腳本**：`src/run_hw3_3.py`  
**環境**：Gridworld 4×4（random mode），三組 seed（0/1/2）重複訓練  
**比較方法**：

| 方法 | 說明 |
|------|------|
| Lightning w/o Tips | 標準 Lightning DQN，固定學習率 |
| Lightning w/ Tips | 加入 CosineAnnealingLR + Gradient Clipping（clip_val=1.0） |

**報告**：[reports/hw3-3_report.md](reports/hw3-3_report.md)

> 可設定環境變數 `HW3_3_TIPS_ONLY=1` 僅重新執行 Tips 版本（跳過 Baseline）：
> ```bash
> HW3_3_TIPS_ONLY=1 python src/run_hw3_3.py
> ```

---

### HW3-4：Rainbow DQN
**執行腳本**：`src/run_hw3_4.py`  
**環境**：Gridworld 4×4（random mode）  
**比較方法**：

| 方法 | 說明 |
|------|------|
| Baseline (DQN + Replay) | 標準對照 |
| Rainbow DQN | Double + Dueling + NoisyNet + PER + Multi-step（n=3） |

**報告**：[reports/hw3-4_report.md](reports/hw3-4_report.md)

---

## 環境安裝

建議使用 Python 3.10+，建立虛擬環境後安裝依賴：

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**依賴套件**（`requirements.txt`）：

```
torch
numpy
pandas
matplotlib
imageio
pillow
pytorch-lightning
```

---

## 執行方式

```bash
# HW3-1：Naive DQN vs Replay
python src/run_hw3_1.py

# HW3-2：Double / Dueling DQN
python src/run_hw3_2.py

# HW3-3：PyTorch Lightning DQN（完整）
python src/run_hw3_3.py

# HW3-3：僅執行 Tips 版本
HW3_3_TIPS_ONLY=1 python src/run_hw3_3.py

# HW3-4：Rainbow DQN
python src/run_hw3_4.py
```

---

## 輸出說明

每個實驗目錄下會產生以下檔案：

| 檔案 | 說明 |
|------|------|
| `metrics.csv` | Episode-level 指標（reward、loss、steps、success） |
| `loss.png` | Loss 曲線圖 |
| `dashboard.gif` | 訓練後 greedy policy 策略動畫 |
| `dashboards/episode_N.gif` | 各訓練階段策略動畫（每 50 回合） |
| `config.json` | 實驗超參數與環境設定 |
| `checkpoints/` | 模型權重 `.pt` 檔案 |

`outputs/summaries/` 下的跨實驗匯總：

| 檔案 | 說明 |
|------|------|
| `loss_compare_hw3-N.png` | 多方法 Loss 曲線比較圖 |
| `metrics_table_hw3-N.csv` | 最後 10 回合平均指標匯總 |


