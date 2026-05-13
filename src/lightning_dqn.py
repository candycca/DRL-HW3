from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from Gridworld import Gridworld
from dqn_common import ACTIONS, QNetwork, TrainConfig, get_state, set_seed, is_converged
from utils.replay_buffer import ReplayBuffer
from utils.visualize import render_board, make_dashboard_gif


class LightningDQN(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        seed: int,
        mode: str,
        use_scheduler: bool = False,
        clip_val: float | None = None,
        device_name: str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.seed = seed
        self.mode = mode
        self.use_scheduler = use_scheduler
        self.clip_val = clip_val
        self.device_name = device_name
        self.epsilon = config.epsilon_start
        self.loss_fn = nn.SmoothL1Loss()
        self.automatic_optimization = False
        set_seed(seed)

        dummy_env = Gridworld(size=config.board_size, mode=mode)
        state_dim = get_state(dummy_env).shape[0]
        self.q_net = QNetwork(state_dim, len(ACTIONS), config.hidden_dim)
        self.target_net = QNetwork(state_dim, len(ACTIONS), config.hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.replay = ReplayBuffer(config.replay_capacity)
        self.step_count = 0
        self.metrics: List[Dict[str, float]] = []
        self.last_metrics: Dict[str, float] = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_episodes,
                eta_min=self.config.lr * 0.1,
            )
            return optimizer
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            list(range(self.config.max_episodes)),
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def training_step(self, batch, batch_idx: int):
        episode = batch_idx + 1
        env = Gridworld(size=self.config.board_size, mode=self.mode)
        state = get_state(env)
        total_reward = 0
        losses: List[float] = []
        steps = 0
        success = 0

        for _ in range(self.config.max_steps):
            action_idx = self.choose_action(state)
            env.makeMove(ACTIONS[action_idx])
            reward = env.reward()
            done = reward in [10, -10]
            next_state = get_state(env)

            self.replay.push((state, action_idx, reward, next_state, done))

            if len(self.replay) >= self.config.batch_size:
                batch_data = self.replay.sample(self.config.batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch_data))

                states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

                q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    next_q = self.target_net(next_states_t).max(1)[0]
                    targets = rewards_t + (1 - dones_t) * self.config.gamma * next_q

                loss = self.loss_fn(q_values, targets)
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_val)
                opt.step()
                if self.use_scheduler and hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.step()
                losses.append(float(loss.item()))

            total_reward += reward
            steps += 1
            self.step_count += 1
            if reward == 10:
                success = 1
            if done:
                break
            state = next_state

            if self.step_count % self.config.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        self.last_metrics = {
            "episode": episode,
            "reward": total_reward,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "steps": steps,
            "success": success,
        }
        self.metrics.append(self.last_metrics)
        self.log_dict(self.last_metrics, prog_bar=False, logger=False)


        if is_converged(self.metrics, self.config):
            self.trainer.should_stop = True

    def make_dashboard(self, output_path: Path) -> None:
        env = Gridworld(size=self.config.board_size, mode=self.mode)
        frames = []
        state = get_state(env)
        for _ in range(self.config.max_steps):
            frames.append(render_board(env.display()))
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_idx = int(torch.argmax(self.q_net(state_t), dim=1).item())
            env.makeMove(ACTIONS[action_idx])
            reward = env.reward()
            if reward in [10, -10]:
                frames.append(render_board(env.display()))
                break
            state = get_state(env)
        make_dashboard_gif(frames, str(output_path))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def save_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self.config), f, indent=2)
