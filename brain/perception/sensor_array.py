"""
brain/perception/sensor_array.py

Simulates a ring of direction-tuned proximity sensors around the robot
- like insect ommatidia or a ring of IR/ultrasonic rangefinders, not a
camera. Each sensor has a PREFERRED DIRECTION and responds most
strongly when an object is close and in that direction (cosine tuning,
the standard model for directionally-tuned biological receptors).

This is the honest "what a real sensor would produce" step: no real
hardware is available, so this generates synthetic-but-principled
analog readings from a known object position, which then feed the SNN
(snn_layer.py) as input current - exactly the role real transduced
sensor signals would play.
"""
from __future__ import annotations

import numpy as np


class DirectionTunedSensorArray:
    def __init__(self, n_sensors: int = 12, max_range: float = 10.0):
        self.n_sensors = n_sensors
        self.max_range = max_range
        # Preferred direction of each sensor, evenly spaced around the robot -
        # like a ring of ultrasonic sensors on a real rover chassis.
        angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
        self.preferred_directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_sensors, 2)

    def sense(self, object_relative_position: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        """object_relative_position: (2,) vector from robot to object.
        Returns (n_sensors,) analog readings in [0, 1] - each sensor's
        response is cosine-tuned to direction AND falls off with
        distance, matching how a real directional proximity sensor
        (or a population of direction-selective neurons) behaves."""
        distance = np.linalg.norm(object_relative_position)
        if distance < 1e-6:
            direction = np.zeros(2)
        else:
            direction = object_relative_position / distance

        proximity = max(0.0, 1.0 - distance / self.max_range)  # 1.0 = touching, 0.0 = out of range

        cosine_tuning = self.preferred_directions @ direction  # (n_sensors,), in [-1, 1]
        cosine_tuning = np.clip(cosine_tuning, 0.0, None)      # sensors facing away read zero, not negative

        readings = cosine_tuning * proximity
        if noise_std > 0:
            rng = np.random.default_rng()
            readings = np.clip(readings + rng.normal(0, noise_std, self.n_sensors), 0.0, 1.0)
        return readings