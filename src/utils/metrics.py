from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import pandas as pd

@dataclass
class MetricSummary:
    mean_reward: float
    success_rate: float
    mean_steps: float
    mean_loss: float


def aggregate_metrics(metric_paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = [pd.read_csv(path) for path in metric_paths]
    if not frames:
        raise ValueError("No metric files provided for aggregation")
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined.groupby("episode", as_index=False).mean(numeric_only=True)


def summarize_metrics(metrics: pd.DataFrame, window: int = 10) -> MetricSummary:
    tail = metrics.tail(window)
    return MetricSummary(
        mean_reward=float(tail["reward"].mean()),
        success_rate=float(tail["success"].mean()),
        mean_steps=float(tail["steps"].mean()),
        mean_loss=float(tail["loss"].mean()),
    )


def metrics_to_table(rows: Dict[str, MetricSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": name,
                "mean_reward": summary.mean_reward,
                "success_rate": summary.success_rate,
                "mean_steps": summary.mean_steps,
                "mean_loss": summary.mean_loss,
            }
            for name, summary in rows.items()
        ]
    )
