"""
environment/multi_robot_env.py

Multi-robot coordination environment: wraps SwarmManager in a
gym-like step/reset interface where EACH agent gets its own action
(discrete move direction) and its own local reward, but they share one
world/stigmergy map - this is what training/multi_agent_trainer.py
trains against.
"""
from __future__ import annotations

import numpy as np

from simulation.swarm.swarm_manager import SwarmManager
from utils.config import SwarmConfig


class MultiRobotEnv:
    ACTIONS = {0: np.zeros(2), 1: np.array([0, 1.0]), 2: np.array([0, -1.0]),
               3: np.array([-1.0, 0]), 4: np.array([1.0, 0])}

    def __init__(self, config: SwarmConfig | None = None, world_size: tuple = (30, 30),
                 target_pos: np.ndarray | None = None, max_steps: int = 150,
                 seed_stream: str = "multi_robot_env"):
        self.config = config or SwarmConfig()
        self.world_size = world_size
        self.target_pos = target_pos if target_pos is not None else np.array(world_size) / 2
        self.max_steps = max_steps
        self.seed_stream = seed_stream
        self._permanently_dead: set = set()  # survives across reset() - see kill_agent()
        self.reset()

    def kill_agent(self, agent_id: int) -> None:
        """Permanently destroys a robot. Unlike an RL episode boundary
        (which is just a training-convenience reset), this persists
        across reset() calls - a destroyed physical robot doesn't come
        back because the training loop started a new episode."""
        self._permanently_dead.add(agent_id)
        for a in self.swarm.agents:
            if a.agent_id == agent_id:
                a.alive = False

    def reset(self):
        self.swarm = SwarmManager(config=self.config, world_size=self.world_size, seed_stream=self.seed_stream)
        for a in self.swarm.agents:
            if a.agent_id in self._permanently_dead:
                a.alive = False
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> dict:
        """Per-agent local observation: own pos/vel + relative target position."""
        obs = {}
        for a in self.swarm.agents:
            if not a.alive:
                continue
            rel_target = self.target_pos - a.position
            obs[a.agent_id] = np.concatenate([a.position, a.velocity, rel_target])
        return obs

    def step(self, agent_actions: dict):
        """agent_actions: {agent_id: discrete_action}. Applies a direct
        velocity nudge per agent (on top of the swarm's own boids
        behavior) representing the learned policy's steering input,
        then lets SwarmManager.step() run flocking + stigmergy + fault
        tolerance as usual."""
        for a in self.swarm.agents:
            if a.alive and a.agent_id in agent_actions:
                nudge = self.ACTIONS[int(agent_actions[a.agent_id])]
                a.velocity += nudge * 0.5

        info = self.swarm.step()
        self.step_count += 1

        rewards = {}
        for a in self.swarm.agents:
            if not a.alive:
                continue
            dist = np.linalg.norm(a.position - self.target_pos)
            rewards[a.agent_id] = -dist * 0.01

        terminated = bool(np.linalg.norm(info["centroid"] - self.target_pos) < 1.0)
        truncated = bool(self.step_count >= self.max_steps)

        return self._get_obs(), rewards, terminated, truncated, info