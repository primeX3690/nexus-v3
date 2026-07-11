"""
Shared numeric primitives. Pure numpy - no autodiff library is installed
in this environment, so PPO's network in brain/rl/ppo_agent.py implements
its own forward/backward pass and relies on these helpers for numerically
stable ops (softmax) and standard RL preprocessing (running normalization,
grad clipping).
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtracts max before exponentiating
    to avoid overflow on large logits)."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance normalization over the full array."""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> dict[str, np.ndarray]:
    """Global-norm gradient clipping across a dict of param-name -> grad
    arrays, mirroring torch.nn.utils.clip_grad_norm_ semantics but for
    plain numpy grads (used by the hand-rolled PPO backward pass)."""
    total_sq = sum(float(np.sum(g ** 2)) for g in grads.values())
    total_norm = float(np.sqrt(total_sq))
    if total_norm <= max_norm or total_norm == 0.0:
        return grads
    scale = max_norm / (total_norm + 1e-6)
    return {k: g * scale for k, g in grads.items()}


class RunningMeanStd:
    """Welford's online algorithm for tracking mean/variance of a stream
    of observations. Used to normalize PPO observations/rewards without
    needing the full history in memory."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # avoid div-by-zero on first update

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + eps)


def running_mean_std(shape: tuple[int, ...] = ()) -> RunningMeanStd:
    return RunningMeanStd(shape)