import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.dqn_common import TrainConfig
from src.dqn_replay import DQNReplayTrainer
from src.rainbow_dqn import RainbowDQNTrainer


def _config():
    return TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)


def test_baseline_random_smoke():
    trainer = DQNReplayTrainer(config=_config(), seed=0, mode="random")
    metrics = trainer.train()
    assert metrics


def test_rainbow_smoke():
    trainer = RainbowDQNTrainer(config=_config(), seed=0, mode="random")
    metrics = trainer.train()
    assert metrics
