import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.dqn_common import TrainConfig
from src.lightning_dqn import LightningDQN
import pytorch_lightning as pl


def test_lightning_baseline_smoke():
    config = TrainConfig(max_episodes=4, min_episodes=2, converge_window=2, max_steps=5, replay_capacity=50, batch_size=4)
    model = LightningDQN(config=config, seed=0, mode="random", use_scheduler=False)
    trainer = pl.Trainer(max_epochs=config.max_episodes, enable_progress_bar=False, logger=False)
    trainer.fit(model)
    assert model.metrics
