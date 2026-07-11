"""
environment/gym_wrapper.py

A minimal Gymnasium-compatible environment for a single point-mass
robot navigating to a target in 2D, with simple obstacle penalties.
This is intentionally simple (no PyBullet dependency - see
pybullet_env.py for why) so it's fast enough to run thousands of steps
per second on an 8GB CPU-only laptop, which matters for PPO's on-policy
sample requirements.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, world_size: float = 20.0, n_obstacles: int = 3,
                 max_steps: int = 200, seed: int | None = None):
        super().__init__()
        self.world_size = world_size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps

        # observation: [robot_x, robot_y, robot_vx, robot_vy, target_x, target_y, *obstacle_xy...]
        obs_dim = 6 + 2 * n_obstacles
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        # discrete actions: 0=stay,1=up,2=down,3=left,4=right
        self.action_space = spaces.Discrete(5)
        self._action_to_accel = {
            0: np.array([0.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]), 4: np.array([1.0, 0.0]),
        }

        self._np_random = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.robot_pos = self._np_random.uniform(0, self.world_size, size=2)
        self.robot_vel = np.zeros(2)
        self.target_pos = self._np_random.uniform(0, self.world_size, size=2)
        self.obstacles = self._np_random.uniform(0, self.world_size, size=(self.n_obstacles, 2))
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.robot_pos, self.robot_vel, self.target_pos, self.obstacles.flatten()])

    def step(self, action: int):
        accel = self._action_to_accel[int(action)]
        self.robot_vel = 0.8 * self.robot_vel + accel  # damped acceleration
        self.robot_pos = np.clip(self.robot_pos + self.robot_vel, 0, self.world_size)
        self.step_count += 1

        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        reward = -dist_to_target * 0.01  # dense shaping: closer is better

        obstacle_penalty = 0.0
        for obs_pos in self.obstacles:
            d = np.linalg.norm(self.robot_pos - obs_pos)
            if d < 1.0:
                obstacle_penalty += (1.0 - d) * 2.0
        reward -= obstacle_penalty

        terminated = bool(dist_to_target < 0.5)
        if terminated:
            reward += 10.0
        truncated = bool(self.step_count >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, {"dist_to_target": dist_to_target}