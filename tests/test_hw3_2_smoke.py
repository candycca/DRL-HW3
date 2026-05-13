import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.dqn_common import TrainConfig
from src.dqn_replay import DQNReplayTrainer
from src.double_dqn import DoubleDQNTrainer
from src.dueling_dqn import DuelingDQNTrainer
from src.dueling_double_dqn import DuelingDoubleDQNTrainer


def _config():
    return TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)


def test_baseline_player_smoke():
    trainer = DQNReplayTrainer(config=_config(), seed=0, mode="player")
    metrics = trainer.train()
    assert metrics


def test_double_smoke():
    trainer = DoubleDQNTrainer(config=_config(), seed=0, mode="player")
    metrics = trainer.train()
    assert metrics


def test_dueling_smoke():
    trainer = DuelingDQNTrainer(config=_config(), seed=0, mode="player")
    metrics = trainer.train()
    assert metrics


def test_dueling_double_smoke():
    trainer = DuelingDoubleDQNTrainer(config=_config(), seed=0, mode="player")
    metrics = trainer.train()
    assert metrics
