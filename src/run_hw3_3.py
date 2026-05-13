from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from dqn_common import TrainConfig
from lightning_dqn import LightningDQN
from utils.metrics import aggregate_metrics, metrics_to_table, summarize_metrics
from utils.visualize import plot_loss, plot_loss_compare


EXPERIMENTS = {
    "lightning_baseline_random": {"use_scheduler": False, "clip_val": None},
    "lightning_tips_random": {"use_scheduler": True, "clip_val": 1.0},
}

RUN_TIPS_ONLY = os.getenv("HW3_3_TIPS_ONLY", "0") == "1"


def run_experiment(name: str, config: TrainConfig, seeds: List[int], base_dir: Path, use_scheduler: bool, clip_val: float | None) -> Path:
    exp_dir = base_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dashboards_dir = exp_dir / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = []
    first_model: LightningDQN | None = None

    for seed in seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model = LightningDQN(
            config=config,
            seed=seed,
            mode="random",
            use_scheduler=use_scheduler,
            clip_val=clip_val,
        )
        trainer = pl.Trainer(
            max_epochs=config.max_episodes,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model)

        metrics = model.metrics
        metrics_path = seed_dir / "metrics.csv"
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        plot_loss(pd.DataFrame(metrics), str(seed_dir / "loss.png"), title=f"{name} seed {seed}")
        model.save_checkpoint(exp_dir / "checkpoints" / f"{name}_seed_{seed}.pt")
        metric_paths.append(str(metrics_path))
        if first_model is None:
            first_model = model

        for row in metrics:
            episode = int(row["episode"])
            if episode % config.dashboard_interval == 0:
                model.make_dashboard(dashboards_dir / f"episode_{episode}.gif")

    aggregated = aggregate_metrics(metric_paths)
    aggregated.to_csv(exp_dir / "metrics.csv", index=False)
    plot_loss(aggregated, str(exp_dir / "loss.png"), title=name)
    if first_model:
        first_model.make_dashboard(exp_dir / "dashboard.gif")
        first_model.save_config(exp_dir / "config.json")
    return exp_dir


def main():
    base_output = Path("outputs/hw3-3")
    summary_dir = Path("outputs/summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig()
    seeds = [0, 1, 2]

    exp_dirs: Dict[str, Path] = {}
    experiments = {"lightning_tips_random": EXPERIMENTS["lightning_tips_random"]} if RUN_TIPS_ONLY else EXPERIMENTS
    for name, exp in experiments.items():
        exp_dirs[name] = run_experiment(
            name,
            config,
            seeds,
            base_output,
            exp["use_scheduler"],
            exp["clip_val"],
        )

    compare_data = {}
    summary_rows = {}
    for name, exp_dir in exp_dirs.items():
        metrics = pd.read_csv(exp_dir / "metrics.csv")
        compare_data[name] = metrics
        summary_rows[name] = summarize_metrics(metrics)

    plot_loss_compare(compare_data, str(summary_dir / "loss_compare_hw3-3.png"))
    metrics_table = metrics_to_table(summary_rows)
    metrics_table.to_csv(summary_dir / "metrics_table_hw3-3.csv", index=False)

    with (summary_dir / "config_hw3-3.json").open("w") as f:
        json.dump(config.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
