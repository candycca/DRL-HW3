from __future__ import annotations

from typing import Callable, Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from Gridworld import Gridworld
from dqn_common import ACTIONS, DuelingQNetwork, TrainConfig, get_state, set_seed, is_converged
from utils.replay_buffer import ReplayBuffer
from utils.visualize import render_board, make_dashboard_gif


class DuelingDQNTrainer:
    def __init__(self, config: TrainConfig, seed: int, mode: str, device: str = "cpu"):
        self.config = config
        self.seed = seed
        self.mode = mode
        self.device = device
        self.epsilon = config.epsilon_start
        self.loss_fn = nn.SmoothL1Loss()
        set_seed(seed)

        dummy_env = Gridworld(size=config.board_size, mode=mode)
        state_dim = get_state(dummy_env).shape[0]
        self.q_net = DuelingQNetwork(state_dim, len(ACTIONS), config.hidden_dim).to(device)
        self.target_net = DuelingQNetwork(state_dim, len(ACTIONS), config.hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.replay_capacity)

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train(self, on_episode_end: Optional[Callable[[int, "DuelingDQNTrainer"], None]] = None) -> List[Dict[str, float]]:
        metrics = []
        step_count = 0
        for episode in range(1, self.config.max_episodes + 1):
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
                    batch = self.replay.sample(self.config.batch_size)
                    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

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
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss.item()))

                total_reward += reward
                steps += 1
                step_count += 1
                if reward == 10:
                    success = 1
                if done:
                    break
                state = next_state

                if step_count % self.config.target_update == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

            metrics.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "loss": float(np.mean(losses)) if losses else 0.0,
                    "steps": steps,
                    "success": success,
                }
            )
            if on_episode_end:
                on_episode_end(episode, self)
            if is_converged(metrics, self.config):
                break
        return metrics

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def save_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.config.__dict__, f, indent=2)

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
