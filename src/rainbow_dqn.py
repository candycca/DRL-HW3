from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from Gridworld import Gridworld
from dqn_common import ACTIONS, TrainConfig, get_state, set_seed, is_converged
from utils.noisy_linear import NoisyLinear
from utils.per_buffer import PrioritizedReplayBuffer
from utils.visualize import render_board, make_dashboard_gif


class RainbowQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, std_init: float):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, std_init),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init),
            nn.ReLU(),
            NoisyLinear(hidden_dim, output_dim, std_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowDQNTrainer:
    def __init__(self, config: TrainConfig, seed: int, mode: str, device: str = "cpu"):
        self.config = config
        self.seed = seed
        self.mode = mode
        self.device = device
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        set_seed(seed)

        dummy_env = Gridworld(size=config.board_size, mode=mode)
        state_dim = get_state(dummy_env).shape[0]
        self.q_net = RainbowQNetwork(state_dim, len(ACTIONS), config.hidden_dim, config.noisy_std_init).to(device)
        self.target_net = RainbowQNetwork(state_dim, len(ACTIONS), config.hidden_dim, config.noisy_std_init).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = PrioritizedReplayBuffer(config.replay_capacity, alpha=config.per_alpha, eps=config.per_eps)
        self.n_step_buffer: Deque = deque(maxlen=config.n_step)
        self.step_count = 0

    def choose_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def _get_n_step_transition(self) -> Optional[tuple]:
        if len(self.n_step_buffer) < self.config.n_step:
            return None
        reward, next_state, done = 0.0, None, False
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.config.gamma**idx) * r
            next_state = ns
            done = d
            if d:
                break
        state, action, _, _, _ = self.n_step_buffer[0]
        return state, action, reward, next_state, done

    def train(self, on_episode_end: Optional[Callable[[int, "RainbowDQNTrainer"], None]] = None) -> List[Dict[str, float]]:
        metrics = []
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

                self.n_step_buffer.append((state, action_idx, reward, next_state, done))
                n_step_transition = self._get_n_step_transition()
                if n_step_transition:
                    self.replay.push(n_step_transition)

                if len(self.replay) >= self.config.batch_size:
                    batch = self.replay.sample(self.config.batch_size, beta=self.config.per_beta)
                    states, actions, rewards, next_states, dones = map(np.array, zip(*batch.transitions))

                    states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
                    actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                    dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
                    weights_t = torch.tensor(batch.weights, dtype=torch.float32, device=self.device)

                    q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)
                    with torch.no_grad():
                        next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                        next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
                        targets = rewards_t + (1 - dones_t) * (self.config.gamma**self.config.n_step) * next_q

                    td_errors = targets - q_values
                    loss = (self.loss_fn(q_values, targets) * weights_t).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss.item()))

                    self.replay.update_priorities(batch.indices, np.abs(td_errors.detach().cpu().numpy()))
                    self.q_net.reset_noise()
                    self.target_net.reset_noise()

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
