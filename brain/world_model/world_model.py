"""
brain/world_model/world_model.py

Wraps MambaSSM with a linear prediction head so the brain can "imagine"
future latent states without acting in the real environment - this is
what the MCTS planner (brain/reasoning/planner.py) rolls out against
instead of the real, expensive simulator.

WorldModel.imagine(state, action, n_steps) repeatedly:
  1. encodes (state, action) into the SSM's input space
  2. steps the SSM forward
  3. decodes the SSM output back into a predicted next state
without ever touching the real environment.
"""
from __future__ import annotations

import numpy as np

from utils.config import MambaConfig
from utils.seeding import get_rng
from brain.world_model.mamba_ssm import MambaSSM


class WorldModel:
    def __init__(self, state_dim: int, action_dim: int, config: MambaConfig | None = None,
                 seed_stream: str = "world_model"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or MambaConfig()
        rng = get_rng(seed_stream)

        self.ssm = MambaSSM(self.config, seed_stream=f"{seed_stream}_ssm")

        in_dim = state_dim + action_dim
        d_model = self.config.d_model
        scale_in = 1.0 / np.sqrt(in_dim)
        scale_out = 1.0 / np.sqrt(d_model)
        self.W_encode = rng.normal(0, scale_in, size=(in_dim, d_model))
        self.b_encode = np.zeros(d_model)
        self.W_decode = rng.normal(0, scale_out, size=(d_model, state_dim))
        self.b_decode = np.zeros(state_dim)

    def reset(self) -> None:
        self.ssm.reset_state()

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One imagined step: (state, action) -> predicted next state.
        Does NOT touch the real environment."""
        sa = np.concatenate([state, action])
        encoded = np.tanh(sa @ self.W_encode + self.b_encode)
        ssm_out = self.ssm.step(encoded)
        delta_state = ssm_out @ self.W_decode + self.b_decode
        return state + delta_state  # residual prediction: predict the CHANGE in state

    def imagine(self, state: np.ndarray, action_sequence: list[np.ndarray], reset: bool = True) -> np.ndarray:
        """Roll forward a sequence of actions purely in imagination.
        Returns (len(action_sequence)+1, state_dim): [state_0, state_1, ..., state_T]."""
        if reset:
            self.reset()
        states = [state]
        cur = state
        for action in action_sequence:
            cur = self.predict_next(cur, action)
            states.append(cur)
        return np.stack(states)