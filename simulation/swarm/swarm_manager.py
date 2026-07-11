"""
simulation/swarm/swarm_manager.py

Orchestrates a full swarm: steps all agents each tick, maintains the
shared stigmergy map (deposit + evaporate), runs each agent's fault
tolerance heartbeat check, and exposes swarm-level statistics
(centroid, spread, alive count) that the genetic engine can use as a
fitness signal when evolving swarm behaviors (e.g. evolving the boids
weight dict itself, not just code).
"""
from __future__ import annotations

import numpy as np

from simulation.swarm.swarm_agent import SwarmAgent
from simulation.swarm.stigmergy_map import StigmergyMap
from simulation.swarm.fault_tolerance import FaultToleranceManager
from utils.config import SwarmConfig
from utils.seeding import get_rng


class SwarmManager:
    def __init__(self, config: SwarmConfig | None = None, world_size: tuple = (50, 50),
                 seed_stream: str = "swarm_manager"):
        self.config = config or SwarmConfig()
        self.world_size = world_size
        rng = get_rng(seed_stream)

        self.agents: list[SwarmAgent] = []
        self.fault_managers: dict[int, FaultToleranceManager] = {}
        for i in range(self.config.n_agents):
            pos = rng.uniform(0, min(world_size), size=2)
            self.agents.append(SwarmAgent(agent_id=i, position=pos, perception_radius=self.config.comm_range))
            self.fault_managers[i] = FaultToleranceManager(
                agent_id=i,
                heartbeat_interval_steps=self.config.heartbeat_interval_steps,
                heartbeat_timeout_steps=self.config.heartbeat_timeout_steps,
            )

        self.stigmergy = StigmergyMap(width=world_size[0], height=world_size[1])
        self.step_count = 0

    def kill_agent(self, agent_id: int) -> None:
        for a in self.agents:
            if a.agent_id == agent_id:
                a.alive = False

    def step(self, weights: dict | None = None) -> dict:
        all_ids = {a.agent_id for a in self.agents}
        alive_agents = [a for a in self.agents if a.alive]

        # heartbeat exchange: every alive agent that should broadcast this
        # step is "heard" by every other alive agent's fault manager.
        for a in alive_agents:
            fm = self.fault_managers[a.agent_id]
            if fm.should_broadcast(self.step_count):
                for other in alive_agents:
                    if other.agent_id != a.agent_id:
                        self.fault_managers[other.agent_id].receive_heartbeat(a.agent_id, self.step_count)

        newly_dead_by_agent = {}
        for a in alive_agents:
            fm = self.fault_managers[a.agent_id]
            newly_dead = fm.tick(self.step_count, all_ids)
            if newly_dead:
                newly_dead_by_agent[a.agent_id] = newly_dead

        for a in alive_agents:
            a.step(alive_agents, stigmergy=self.stigmergy, weights=weights, world_size=self.world_size)
            gx, gy = int(np.clip(a.position[0], 0, self.world_size[0] - 1)), \
                     int(np.clip(a.position[1], 0, self.world_size[1] - 1))
            self.stigmergy.deposit(gx, gy, amount=0.5)

        self.stigmergy.evaporate()
        self.step_count += 1

        return {
            "step": self.step_count,
            "alive_count": len(alive_agents),
            "centroid": self.centroid(),
            "spread": self.spread(),
            "newly_dead_detections": newly_dead_by_agent,
        }

    def centroid(self) -> np.ndarray:
        alive = [a.position for a in self.agents if a.alive]
        return np.mean(alive, axis=0) if alive else np.zeros(2)

    def spread(self) -> float:
        alive = [a.position for a in self.agents if a.alive]
        if len(alive) < 2:
            return 0.0
        c = np.mean(alive, axis=0)
        return float(np.mean([np.linalg.norm(p - c) for p in alive]))