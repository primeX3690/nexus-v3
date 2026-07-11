"""
simulation/swarm/stigmergy_map.py

Digital pheromone grid: agents deposit "pheromone" at grid cells (e.g.
marking "explored" or "found resource here"), and the field decays over
time (evaporation). This is indirect coordination (stigmergy) - agents
never talk to each other directly, they just read/write a shared
environment, which scales far better than pairwise messaging for large
swarms and degrades gracefully if agents drop out.
"""
from __future__ import annotations

import numpy as np


class StigmergyMap:
    def __init__(self, width: int, height: int, decay_rate: float = 0.02, n_channels: int = 1):
        """n_channels lets you track multiple pheromone types (e.g.
        'explored' vs 'danger') on the same grid independently."""
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.n_channels = n_channels
        self.grid = np.zeros((height, width, n_channels), dtype=np.float64)

    def _clip_coords(self, x: int, y: int) -> tuple[int, int]:
        return int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))

    def deposit(self, x: int, y: int, amount: float = 1.0, channel: int = 0) -> None:
        cx, cy = self._clip_coords(x, y)
        self.grid[cy, cx, channel] += amount

    def read(self, x: int, y: int, channel: int = 0) -> float:
        cx, cy = self._clip_coords(x, y)
        return float(self.grid[cy, cx, channel])

    def read_neighborhood(self, x: int, y: int, radius: int = 1, channel: int = 0) -> np.ndarray:
        cx, cy = self._clip_coords(x, y)
        y0, y1 = max(0, cy - radius), min(self.height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(self.width, cx + radius + 1)
        return self.grid[y0:y1, x0:x1, channel]

    def gradient_direction(self, x: int, y: int, channel: int = 0) -> np.ndarray:
        """Returns a unit-ish (dx, dy) pointing toward the strongest
        nearby pheromone concentration - what a 'follow the trail' agent
        policy would use. Returns (0,0) if the neighborhood is flat."""
        neighborhood = self.read_neighborhood(x, y, radius=1, channel=channel)
        if neighborhood.size == 0 or np.allclose(neighborhood, neighborhood.flat[0]):
            return np.zeros(2)
        cy_local, cx_local = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
        # local center is wherever (x,y) sits within the clipped window - recompute directly
        cx, cy = self._clip_coords(x, y)
        y0 = max(0, cy - 1)
        x0 = max(0, cx - 1)
        target_x, target_y = x0 + cx_local, y0 + cy_local
        direction = np.array([target_x - cx, target_y - cy], dtype=np.float64)
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0 else np.zeros(2)

    def evaporate(self) -> None:
        self.grid *= (1.0 - self.decay_rate)
        self.grid[self.grid < 1e-6] = 0.0  # avoid denormal float drift forever