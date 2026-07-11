"""
simulation/swarm/swarm_agent.py

An individual swarm member: 2D position/velocity, classic Boids rules
(separation, alignment, cohesion) blended with a pull toward the
strongest nearby stigmergy signal (see stigmergy_map.py) so the swarm
both stays coherent AND is drawn toward discovered resources/targets.
"""
from __future__ import annotations

import numpy as np

from simulation.swarm.stigmergy_map import StigmergyMap


class SwarmAgent:
    def __init__(self, agent_id: int, position: np.ndarray, max_speed: float = 2.0,
                 perception_radius: float = 5.0):
        self.agent_id = agent_id
        self.position = position.astype(np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.max_speed = max_speed
        self.perception_radius = perception_radius
        self.alive = True

    def _neighbors(self, all_agents: list) -> list:
        out = []
        for other in all_agents:
            if other.agent_id == self.agent_id or not other.alive:
                continue
            if np.linalg.norm(other.position - self.position) <= self.perception_radius:
                out.append(other)
        return out

    def _boids_forces(self, neighbors: list) -> tuple:
        if not neighbors:
            return np.zeros(2), np.zeros(2), np.zeros(2)

        positions = np.array([n.position for n in neighbors])
        velocities = np.array([n.velocity for n in neighbors])

        # Cohesion: steer toward the average position of neighbors.
        cohesion = positions.mean(axis=0) - self.position

        # Alignment: steer toward the average heading of neighbors.
        alignment = velocities.mean(axis=0) - self.velocity

        # Separation: steer away from neighbors that are too close,
        # weighted inversely by distance so very close neighbors dominate.
        separation = np.zeros(2)
        for n in neighbors:
            diff = self.position - n.position
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                separation += diff / (dist ** 2)

        return cohesion, alignment, separation

    def step(self, all_agents: list, stigmergy: StigmergyMap | None = None,
              weights: dict | None = None, dt: float = 1.0, world_size: tuple | None = None) -> None:
        if not self.alive:
            return
        w = weights or {"cohesion": 0.5, "alignment": 0.7, "separation": 1.2, "stigmergy": 0.8}

        neighbors = self._neighbors(all_agents)
        cohesion, alignment, separation = self._boids_forces(neighbors)

        stigmergy_pull = np.zeros(2)
        if stigmergy is not None:
            gx, gy = int(self.position[0]), int(self.position[1])
            stigmergy_pull = stigmergy.gradient_direction(gx, gy)

        acceleration = (
            w["cohesion"] * cohesion
            + w["alignment"] * alignment
            + w["separation"] * separation
            + w["stigmergy"] * stigmergy_pull
        )

        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        self.position += self.velocity * dt

        if world_size is not None:
            self._reflect_at_boundary(world_size)

    def _reflect_at_boundary(self, world_size: tuple) -> None:
        """Robots operate in a bounded physical area (e.g. a Mars sim
        arena) - without this, unconstrained boids drift outside the
        world indefinitely since cohesion only pulls agents toward each
        other, never back toward the world itself. Reflect velocity and
        clamp position at each edge, like a wall bounce."""
        for dim in range(2):
            if self.position[dim] < 0:
                self.position[dim] = 0
                self.velocity[dim] *= -1
            elif self.position[dim] > world_size[dim]:
                self.position[dim] = world_size[dim]
                self.velocity[dim] *= -1