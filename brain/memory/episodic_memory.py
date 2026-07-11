"""
brain/memory/episodic_memory.py

Long-term storage of complete episodes (trajectories), with nearest-
neighbor retrieval by initial-state similarity - a simple stand-in for
"has something like this happened before, and how did it go". Pure
numpy cosine similarity search; no vector DB needed at this scale
(hundreds to low-thousands of episodes on an 8GB machine).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Episode:
    states: np.ndarray       # (T, state_dim)
    actions: list
    rewards: np.ndarray      # (T,)
    total_reward: float = field(init=False)

    def __post_init__(self):
        self.total_reward = float(np.sum(self.rewards))


class EpisodicMemory:
    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []

    def store(self, states: np.ndarray, actions: list, rewards: np.ndarray) -> None:
        episode = Episode(states=np.array(states), actions=list(actions), rewards=np.array(rewards))
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            # Drop the worst-reward episode rather than the oldest - keeps
            # the memory biased toward informative (especially successful)
            # experience under a fixed budget.
            worst_idx = min(range(len(self.episodes)), key=lambda i: self.episodes[i].total_reward)
            del self.episodes[worst_idx]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def retrieve_similar(self, query_state: np.ndarray, k: int = 5) -> list[Episode]:
        if not self.episodes:
            return []
        sims = [self._cosine_sim(query_state, ep.states[0]) for ep in self.episodes]
        order = np.argsort(sims)[::-1][:k]
        return [self.episodes[i] for i in order]

    def best_episodes(self, k: int = 5) -> list[Episode]:
        return sorted(self.episodes, key=lambda e: e.total_reward, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self.episodes)