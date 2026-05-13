import tempfile
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.dqn_common import TrainConfig
from src.dqn_naive import DQNNaiveTrainer
from src.dqn_replay import DQNReplayTrainer


def test_naive_smoke():
    config = TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)
    trainer = DQNNaiveTrainer(config=config, seed=0, mode="static")
    metrics = trainer.train()
    assert config.min_episodes <= len(metrics) <= config.max_episodes


def test_replay_smoke():
    config = TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)
    trainer = DQNReplayTrainer(config=config, seed=0, mode="static")
    metrics = trainer.train()
    assert config.min_episodes <= len(metrics) <= config.max_episodes


def test_metrics_csv_roundtrip():
    config = TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)
    trainer = DQNNaiveTrainer(config=config, seed=0, mode="static")
    metrics = trainer.train()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "metrics.csv"
        pd.DataFrame(metrics).to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert "reward" in loaded.columns
