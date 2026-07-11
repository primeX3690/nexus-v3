"""
brain/perception/snn_layer.py

Leaky Integrate-and-Fire (LIF) spiking neuron layer, used as the
perception front-end of NEXUS: converts continuous sensor input into
sparse spike trains, which are cheaper to process and closer to
biological perception than dense activations.

LIF dynamics (discrete-time, per neuron i):
    v_i[t] = v_i[t-1] * decay + I_i[t]          (membrane update)
    if v_i[t] >= v_threshold:
        spike_i[t] = 1
        v_i[t] = v_reset                         (hard reset)
    else:
        spike_i[t] = 0
where decay = exp(-dt / tau_mem).

This is a pure-numpy forward pass (no autodiff) since the SNN here is
used as a fixed/hand-tuned feature encoder feeding the Mamba world
model, not trained end-to-end via backprop-through-spikes (which needs
surrogate gradients and is out of scope for the CPU-only budget here).
"""
from __future__ import annotations

import numpy as np

from utils.config import SNNConfig
from utils.seeding import get_rng


class LIFLayer:
    def __init__(self, n_inputs: int, config: SNNConfig | None = None, seed_stream: str = "snn_layer"):
        self.config = config or SNNConfig()
        self.n_inputs = n_inputs
        self.n_neurons = self.config.n_neurons
        rng = get_rng(seed_stream)

        # Xavier-ish init scaled for spiking regime (inputs are ~[0,1] rates)
        limit = np.sqrt(6.0 / (n_inputs + self.n_neurons))
        self.weights = rng.uniform(-limit, limit, size=(n_inputs, self.n_neurons)).astype(np.float64)

        self.decay = float(np.exp(-self.config.dt / self.config.tau_mem))
        self.reset_state()

    def reset_state(self) -> None:
        self.v = np.zeros(self.n_neurons, dtype=np.float64)
        self.refractory_counter = np.zeros(self.n_neurons, dtype=np.int32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single timestep. x: (n_inputs,) input rates/currents.
        Returns spikes: (n_neurons,) binary array."""
        if x.shape != (self.n_inputs,):
            raise ValueError(f"expected input shape ({self.n_inputs},), got {x.shape}")

        current = x @ self.weights  # (n_neurons,)

        not_refractory = self.refractory_counter <= 0
        self.v = np.where(not_refractory, self.v * self.decay + current, self.v)

        spikes = (self.v >= self.config.v_threshold) & not_refractory
        self.v = np.where(spikes, self.config.v_reset, self.v)

        self.refractory_counter = np.where(
            spikes, self.config.refractory_steps,
            np.maximum(self.refractory_counter - 1, 0)
        )
        return spikes.astype(np.float64)

    def run_sequence(self, x_seq: np.ndarray, reset: bool = True) -> np.ndarray:
        """x_seq: (T, n_inputs). Returns spike train (T, n_neurons)."""
        if reset:
            self.reset_state()
        T = x_seq.shape[0]
        out = np.zeros((T, self.n_neurons), dtype=np.float64)
        for t in range(T):
            out[t] = self.step(x_seq[t])
        return out

    def spike_rate(self, x_seq: np.ndarray, reset: bool = True) -> np.ndarray:
        """Convenience: mean firing rate per neuron over a sequence -
        this is what typically feeds forward into the Mamba world model
        as a fixed-size feature vector."""
        spikes = self.run_sequence(x_seq, reset=reset)
        return spikes.mean(axis=0)