"""
brain/rl/replay_buffer.py

On-policy rollout storage for PPO, plus Generalized Advantage Estimation
(GAE-lambda). PPO is on-policy, so this buffer is filled fresh each
update and then cleared - it is not a large off-policy replay memory
like in DQN.

GAE recap (Schulman et al. 2016):
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
computed backwards through the trajectory. Returns-to-go for the value
loss are then R_t = A_t + V(s_t).
"""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, state, action, log_prob, reward, value, done) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.__init__()

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns), both shape (T,).
        `last_value` is V(s_T) for the state AFTER the final stored
        transition (bootstrap value; pass 0.0 if the episode truly ended)."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float64)
        values = np.array(self.values + [last_value], dtype=np.float64)  # (T+1,)
        gae = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values[:T]
        return advantages, returns

    def get_batches(self, batch_size: int, rng: np.random.Generator):
        """Yields shuffled minibatch index arrays covering the full buffer once."""
        n = len(self)
        indices = np.arange(n)
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            yield indices[start:start + batch_size]