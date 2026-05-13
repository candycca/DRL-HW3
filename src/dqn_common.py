from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import random
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["u", "d", "l", "r"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_state(env) -> np.ndarray:
    state = env.board.render_np().astype(np.float32)
    return state.flatten()


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


@dataclass
class TrainConfig:
    max_episodes: int = 1000
    min_episodes: int = 200
    converge_window: int = 30
    reward_std_threshold: float = 1.0
    loss_std_threshold: float = 0.05
    dashboard_interval: int = 50
    num_workers: int = 2
    max_steps: int = 50
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    lr: float = 1e-3
    hidden_dim: int = 128
    batch_size: int = 32
    replay_capacity: int = 2000
    target_update: int = 50
    board_size: int = 4
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_eps: float = 1e-6
    noisy_std_init: float = 0.5


def is_converged(metrics: Sequence[dict], config: TrainConfig) -> bool:
    if len(metrics) < config.min_episodes:
        return False
    window = metrics[-config.converge_window :]
    rewards = np.array([row["reward"] for row in window], dtype=np.float32)
    losses = np.array([row["loss"] for row in window], dtype=np.float32)
    if rewards.size < config.converge_window:
        return False
    reward_std = float(np.std(rewards))
    loss_std = float(np.std(losses))
    return reward_std <= config.reward_std_threshold and loss_std <= config.loss_std_threshold
