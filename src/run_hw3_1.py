from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from dqn_common import TrainConfig
from dqn_naive import DQNNaiveTrainer
from dqn_replay import DQNReplayTrainer
from utils.metrics import aggregate_metrics, metrics_to_table, summarize_metrics
from utils.visualize import plot_loss, plot_loss_compare


EXPERIMENTS = {
    "naive_static": {"trainer": DQNNaiveTrainer, "mode": "static"},
    "replay_static": {"trainer": DQNReplayTrainer, "mode": "static"},
    "replay_random": {"trainer": DQNReplayTrainer, "mode": "random"},
}


def run_experiment(name: str, trainer_cls, mode: str, config: TrainConfig, seeds: List[int], base_dir: Path) -> Path:
    exp_dir = base_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dashboards_dir = exp_dir / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = []
    first_trainer = None

    for seed in seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        trainer = trainer_cls(config=config, seed=seed, mode=mode)

        def on_episode_end(ep: int, tr):
            if ep % config.dashboard_interval == 0:
                tr.make_dashboard(dashboards_dir / f"episode_{ep}.gif")

        metrics = trainer.train(on_episode_end=on_episode_end)
        metrics_path = seed_dir / "metrics.csv"
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        plot_loss(pd.DataFrame(metrics), str(seed_dir / "loss.png"), title=f"{name} seed {seed}")
        trainer.save_checkpoint(exp_dir / "checkpoints" / f"{name}_seed_{seed}.pt")
        metric_paths.append(str(metrics_path))
        if first_trainer is None:
            first_trainer = trainer

    aggregated = aggregate_metrics(metric_paths)
    aggregated.to_csv(exp_dir / "metrics.csv", index=False)
    plot_loss(aggregated, str(exp_dir / "loss.png"), title=name)
    if first_trainer:
        first_trainer.make_dashboard(exp_dir / "dashboard.gif")
        first_trainer.save_config(exp_dir / "config.json")
    return exp_dir


def main():
    base_output = Path("outputs/hw3-1")
    summary_dir = Path("outputs/summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig()
    seeds = [0, 1, 2]

    exp_dirs: Dict[str, Path] = {}
    for name, exp in EXPERIMENTS.items():
        exp_dirs[name] = run_experiment(name, exp["trainer"], exp["mode"], config, seeds, base_output)

    compare_data = {}
    summary_rows = {}
    for name, exp_dir in exp_dirs.items():
        metrics = pd.read_csv(exp_dir / "metrics.csv")
        compare_data[name] = metrics
        summary_rows[name] = summarize_metrics(metrics)

    plot_loss_compare(compare_data, str(summary_dir / "loss_compare_hw3-1.png"))
    metrics_table = metrics_to_table(summary_rows)
    metrics_table.to_csv(summary_dir / "metrics_table_hw3-1.csv", index=False)

    with (summary_dir / "config_hw3-1.json").open("w") as f:
        json.dump(config.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
