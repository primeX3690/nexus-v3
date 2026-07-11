"""
brain/memory/working_memory.py

Short-term context: the last N timesteps of (state, action, reward),
kept as a fixed-size rolling window. This is what feeds the reasoning
layer's "what just happened" context - cheap, no persistence, wiped on
reset.
"""
from __future__ import annotations

from collections import deque
import numpy as np


class WorkingMemory:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action, reward: float) -> None:
        self.buffer.append({"state": state, "action": action, "reward": reward})

    def recent(self, n: int | None = None) -> list[dict]:
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def mean_recent_reward(self, n: int = 10) -> float:
        items = self.recent(n)
        if not items:
            return 0.0
        return float(np.mean([i["reward"] for i in items]))

    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)