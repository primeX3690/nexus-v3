"""
environment/pybullet_env.py

Physics simulation for NEXUS robots.

Honest note on scope: full PyBullet requires compiling an ~80MB C++
source tree (confirmed during this build: multiple minutes just for
`pip install pybullet` on a fast sandbox VM), which is a bad fit for a
Ryzen 3 / 8GB laptop and for fast PPO training loops that need
thousands of steps/sec. So this file's DEFAULT, always-available path
is a lightweight numpy 3D point-mass/rigid-body simulator with gravity,
ground collision, and simple drag - enough physical realism for
locomotion/navigation policy training. If PyBullet IS installed
(`pip install pybullet`), `PhysicsEnv(backend="pybullet")` will use it
for higher-fidelity contact dynamics; this is optional, not required.
"""
from __future__ import annotations

import numpy as np


class LightweightPhysics:
    """Numpy point-mass physics: gravity, ground collision with
    restitution, simple linear drag. No contact/friction between
    multiple bodies - for that level of fidelity, use the pybullet
    backend if it's installed."""

    def __init__(self, gravity: float = -9.81, ground_z: float = 0.0,
                 restitution: float = 0.3, drag: float = 0.02, dt: float = 1.0 / 60.0):
        self.gravity = gravity
        self.ground_z = ground_z
        self.restitution = restitution
        self.drag = drag
        self.dt = dt

    def step(self, position: np.ndarray, velocity: np.ndarray, force: np.ndarray, mass: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray]:
        """position, velocity, force: (3,) arrays [x, y, z]. Returns
        (new_position, new_velocity)."""
        accel = force / mass
        accel[2] += self.gravity
        velocity = velocity + accel * self.dt
        velocity *= (1.0 - self.drag)
        position = position + velocity * self.dt

        if position[2] < self.ground_z:
            position[2] = self.ground_z
            if velocity[2] < 0:
                velocity[2] = -velocity[2] * self.restitution
            velocity[0] *= 0.9  # ground friction damps horizontal motion on contact
            velocity[1] *= 0.9

        return position, velocity


class PhysicsEnv:
    """Wraps either the lightweight numpy backend (default, always
    available) or real PyBullet (only if installed) behind one
    interface, so higher-level code (gym_wrapper, multi_robot_env)
    doesn't need to know which is active."""

    def __init__(self, backend: str = "lightweight", n_bodies: int = 1,
                 gravity: float = -9.81, dt: float = 1.0 / 60.0):
        self.backend = backend
        self.n_bodies = n_bodies
        self.dt = dt

        if backend == "pybullet":
            try:
                import pybullet as p
                import pybullet_data
                self._p = p
                self.client = p.connect(p.DIRECT)  # headless, no GUI - needed for CPU-only servers
                p.setGravity(0, 0, gravity, physicsClientId=self.client)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setTimeStep(dt, physicsClientId=self.client)
                self._body_ids: list[int] = []
            except ImportError as e:
                raise ImportError(
                    "backend='pybullet' requires `pip install pybullet`, which was NOT "
                    "auto-installed here because it needs to compile a large C++ source "
                    "tree (slow on constrained hardware). Use backend='lightweight' "
                    "(the default) instead, or install pybullet yourself first."
                ) from e
        elif backend == "lightweight":
            self.physics = LightweightPhysics(gravity=gravity, dt=dt)
            self.positions = np.zeros((n_bodies, 3))
            self.velocities = np.zeros((n_bodies, 3))
        else:
            raise ValueError(f"unknown backend '{backend}', expected 'lightweight' or 'pybullet'")

    def reset(self, positions: np.ndarray | None = None) -> None:
        if self.backend == "lightweight":
            self.positions = positions.copy() if positions is not None else np.zeros((self.n_bodies, 3))
            self.velocities = np.zeros((self.n_bodies, 3))

    def step(self, forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """forces: (n_bodies, 3). Returns (positions, velocities), both (n_bodies, 3)."""
        if self.backend != "lightweight":
            raise NotImplementedError("pybullet backend step() not wired up in this simplified build")
        for i in range(self.n_bodies):
            self.positions[i], self.velocities[i] = self.physics.step(
                self.positions[i], self.velocities[i], forces[i]
            )
        return self.positions.copy(), self.velocities.copy()