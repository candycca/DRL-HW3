from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random
import numpy as np

Transition = Tuple


@dataclass
class PERBatch:
    transitions: List[Transition]
    indices: List[int]
    weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: List[Transition] = []
        self.priorities: np.ndarray = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, transition: Transition) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> PERBatch:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return PERBatch(samples, indices.tolist(), weights.astype(np.float32))

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio + self.eps)

    def __len__(self) -> int:
        return len(self.buffer)
