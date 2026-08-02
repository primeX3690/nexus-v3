"""
brain/perception/perception_pipeline.py

Closes the perception loop that was previously missing: raw object
position -> synthetic sensor readings (sensor_array.py) -> SNN spike
encoding (snn_layer.py) -> population-vector decoding
(population_decoder.py) -> an estimated relative position the MCTS
planner (or anything else) can actually use, instead of being told the
ground-truth obstacle/goal position directly.

This is the "robot should see, not be told" piece: brain/reasoning/planner.py's
MCTS previously ran against a reward_fn with hardcoded/ground-truth
positions in every demo. PerceptionPipeline lets that same reward_fn
instead depend on what the SNN actually perceived - lossy and
noise-affected, matching real sensing, verified in
tests/test_perception_pipeline.py to correlate with ground truth rather
than match it exactly.
"""
from __future__ import annotations

import numpy as np

from utils.config import SNNConfig
from brain.perception.sensor_array import DirectionTunedSensorArray
from brain.perception.snn_layer import LIFLayer
from brain.perception.population_decoder import PopulationVectorDecoder


class PerceptionPipeline:
    def __init__(self, n_sensors: int = 12, max_range: float = 10.0,
                 snn_config: SNNConfig | None = None, seed_stream: str = "perception_pipeline"):
        self.sensor_array = DirectionTunedSensorArray(n_sensors=n_sensors, max_range=max_range)
        self.snn = LIFLayer(n_inputs=n_sensors, config=snn_config or SNNConfig(n_neurons=n_sensors),
                             seed_stream=f"{seed_stream}_snn")
        self.decoder = PopulationVectorDecoder(n_neurons=self.snn.n_neurons)
        self.max_range = max_range
        self._n_timesteps_for_calibration = 30
        self._calibrate()

    def _encode_and_get_spike_rate(self, relative_position: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        sensor_sequence = np.stack([
            self.sensor_array.sense(relative_position, noise_std=noise_std)
            for _ in range(self._n_timesteps_for_calibration)
        ])
        return self.snn.spike_rate(sensor_sequence)

    def _calibrate(self) -> None:
        """Must run before perceive() - measures each SNN output
        neuron's true preferred direction (see population_decoder.py's
        docstring for why this can't be assumed). Uses noise_std=0 for
        calibration so tuning curves are measured cleanly."""
        self.decoder.calibrate(
            lambda rel_pos: self._encode_and_get_spike_rate(rel_pos, noise_std=0.0)
        )

    def perceive(self, true_relative_position: np.ndarray, n_timesteps: int = 20,
                 sensor_noise_std: float = 0.02) -> dict:
        """Runs the full pipeline for one perception 'glance': generates
        a short sensor reading sequence (SNNs need a time window, not a
        single instant), encodes it, and decodes an estimate using the
        calibrated decoder. Returns the estimate dict plus the raw
        spike rate for anything downstream that wants it directly."""
        sensor_sequence = np.stack([
            self.sensor_array.sense(true_relative_position, noise_std=sensor_noise_std)
            for _ in range(n_timesteps)
        ])
        spike_rate = self.snn.spike_rate(sensor_sequence)
        estimate = self.decoder.decode(spike_rate, max_range=self.max_range)
        estimate["spike_rate"] = spike_rate
        return estimate