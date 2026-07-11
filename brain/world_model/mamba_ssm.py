"""
brain/world_model/mamba_ssm.py

A simplified, Mamba-inspired Selective State Space Model.

Real Mamba (Gu & Dao 2023) uses a hardware-aware parallel scan over a
custom CUDA kernel. None of that applies on a CPU-only Ryzen 3 box, so
this is an honest simplification: a *sequential* scan of the same
selective-SSM recurrence, implemented in plain numpy. It captures the
core idea Mamba is known for - input-dependent (selective) discretization
of a continuous state-space model - without claiming kernel-level
performance parity with the real thing.

Per-timestep recurrence (selective SSM, single-input single-output view
per channel, batched across d_model channels):
    delta_t = softplus(W_delta @ x_t)               (input-dependent step size)
    A_bar_t = exp(delta_t * A)                        (zero-order-hold discretization)
    B_bar_t = delta_t * B_t,  B_t = W_B @ x_t          (selective B)
    h_t     = A_bar_t * h_{t-1} + B_bar_t * x_t
    y_t     = (C_t · h_t) + D * x_t,  C_t = W_C @ x_t  (selective C)

A is a fixed, negative (stable) per-channel-per-state parameter matrix,
initialized with the standard "S4D-real" scheme (negative arithmetic
sequence), which is what keeps the recurrence stable without training.
"""
from __future__ import annotations

import numpy as np

from utils.config import MambaConfig
from utils.seeding import get_rng


def softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable softplus: log(1+exp(x))
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


class MambaSSM:
    def __init__(self, config: MambaConfig | None = None, seed_stream: str = "mamba_ssm"):
        self.config = config or MambaConfig()
        d_model, d_state = self.config.d_model, self.config.d_state
        rng = get_rng(seed_stream)

        # S4D-real init: A[n, s] = -(s+1), broadcast per channel. Negative
        # -> stable (non-exploding) state decay under any positive delta.
        self.A = -np.tile(np.arange(1, d_state + 1, dtype=np.float64), (d_model, 1))  # (d_model, d_state)

        scale = 1.0 / np.sqrt(d_model)
        self.W_delta = rng.normal(0, scale, size=(d_model,)).astype(np.float64)      # per-channel scalar delta proj (simplified)
        self.W_B = rng.normal(0, scale, size=(d_model, d_state)).astype(np.float64)
        self.W_C = rng.normal(0, scale, size=(d_model, d_state)).astype(np.float64)
        self.D = rng.normal(0, scale, size=(d_model,)).astype(np.float64)             # skip connection

        self.h = np.zeros((d_model, d_state), dtype=np.float64)

    def reset_state(self) -> None:
        self.h = np.zeros_like(self.h)

    def step(self, x: np.ndarray) -> np.ndarray:
        """x: (d_model,) -> y: (d_model,). One selective-SSM timestep."""
        d_model, d_state = self.config.d_model, self.config.d_state
        if x.shape != (d_model,):
            raise ValueError(f"expected input shape ({d_model},), got {x.shape}")

        delta = softplus(x * self.W_delta)                       # (d_model,)
        A_bar = np.exp(delta[:, None] * self.A)                  # (d_model, d_state)
        B_t = x[:, None] * self.W_B                               # (d_model, d_state) - selective B
        B_bar = delta[:, None] * B_t                              # (d_model, d_state)

        self.h = A_bar * self.h + B_bar * x[:, None]              # (d_model, d_state)

        C_t = x[:, None] * self.W_C                               # (d_model, d_state) - selective C
        y = np.sum(C_t * self.h, axis=1) + self.D * x             # (d_model,)
        return y

    def run_sequence(self, x_seq: np.ndarray, reset: bool = True) -> np.ndarray:
        """x_seq: (T, d_model) -> y_seq: (T, d_model)."""
        if reset:
            self.reset_state()
        T, d_model = x_seq.shape
        if d_model != self.config.d_model:
            raise ValueError(f"expected d_model={self.config.d_model}, got {d_model}")
        out = np.zeros_like(x_seq)
        for t in range(T):
            out[t] = self.step(x_seq[t])
        return out