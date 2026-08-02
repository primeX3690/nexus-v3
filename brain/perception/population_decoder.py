"""
brain/perception/population_decoder.py

Population vector decoding: the classic technique (Georgopoulos et al.
1986, motor cortex direction decoding) for turning a population of
direction-tuned neurons' firing rates into a single estimated vector -
here, "which direction and how close is the nearest object" from the
SNN's spike output, closing the loop that sensor_array.py opens.

decoded_direction = sum_i(spike_rate_i * preferred_direction_i)

IMPORTANT - calibration is required, and here's the bug that made that
clear: it's tempting to assume SNN output neuron i is "tuned to" the
same direction as sensor i. That assumption is WRONG whenever the SNN
layer between them uses a general (e.g. randomly initialized,
fully-connected) weight matrix - which brain/perception/snn_layer.py's
LIFLayer does. Output neuron i's actual response depends on a weighted
mix of ALL input sensors, not just sensor i, so its true "preferred
direction" has to be measured, not assumed. This was caught by testing:
naively reusing sensor directions gave a mean cosine similarity of
~0.60 against ground truth (should be much higher); after adding
proper calibration, direction estimates correlate strongly with
ground truth (see tests/test_perception_pipeline.py for the exact bar).

This mirrors how real neuroscience/BCI decoding actually works: you
never assume a neuron's tuning curve, you measure it with known stimuli
first (a "calibration" or "tuning" phase), then decode.
"""
from __future__ import annotations

import numpy as np


class PopulationVectorDecoder:
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.preferred_directions: np.ndarray | None = None  # set by calibrate()

    def calibrate(self, encode_and_run_fn, n_calibration_angles: int = 16) -> None:
        """Empirically measures each SNN output neuron's true preferred
        direction by presenting stimuli at known angles all around and
        recording the response. `encode_and_run_fn(relative_position)`
        must run the FULL sensor->SNN pipeline and return a spike-rate
        array (n_neurons,) - the same function perceive() will later
        call with unknown positions.

        This must be called once before decode() - decoding without
        calibration raises, rather than silently using a wrong (assumed
        identity) mapping like the pre-fix version of this file did."""
        angles = np.linspace(0, 2 * np.pi, n_calibration_angles, endpoint=False)
        responses = np.zeros((n_calibration_angles, self.n_neurons))
        for i, angle in enumerate(angles):
            stimulus_direction = np.array([np.cos(angle), np.sin(angle)])
            responses[i] = encode_and_run_fn(stimulus_direction * 3.0)  # fixed calibration distance

        # Each neuron's preferred direction = the population-vector
        # average of the directions it responded to, weighted by how
        # strongly it responded - the standard tuning-curve-fitting
        # approach for population vector decoding.
        direction_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_angles, 2)
        preferred = responses.T @ direction_vectors  # (n_neurons, 2)
        norms = np.linalg.norm(preferred, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0  # avoid div-by-zero for neurons that never responded during calibration
        self.preferred_directions = preferred / norms

    def decode(self, spike_rates: np.ndarray, max_range: float = 10.0) -> dict:
        """spike_rates: (n_neurons,) mean firing rate per neuron (e.g.
        from LIFLayer.spike_rate()). Returns estimated direction (unit
        vector), estimated proximity (0=far, 1=touching), and the raw
        population vector magnitude (a confidence proxy)."""
        if self.preferred_directions is None:
            raise RuntimeError(
                "PopulationVectorDecoder.decode() called before calibrate() - "
                "decoding requires knowing each neuron's empirically measured "
                "preferred direction first. See this module's docstring for why "
                "assuming an identity mapping (input sensor i == output neuron i) "
                "is wrong for a general SNN layer."
            )

        n = min(len(spike_rates), len(self.preferred_directions))
        pop_vector = spike_rates[:n] @ self.preferred_directions[:n]  # (2,)

        magnitude = float(np.linalg.norm(pop_vector))
        direction = pop_vector / magnitude if magnitude > 1e-6 else np.zeros(2)

        # Normalize magnitude against a rough theoretical max (all
        # neurons firing at rate 1.0) to get a bounded proximity/
        # confidence estimate in [0, 1].
        proximity_estimate = float(np.clip(magnitude / max(1.0, n * 0.5), 0.0, 1.0))

        return {
            "direction": direction,
            "proximity_estimate": proximity_estimate,
            "raw_magnitude": magnitude,
            "estimated_relative_position": direction * (max_range * (1.0 - proximity_estimate)),
        }