"""
brain/nexus_brain.py

Master controller. Wires together every brain subsystem into a single
step() call a robot control loop can use:

    sensor_input -> SNN perception -> feature vector
                 -> working memory (context)
                 -> symbolic safety check (hard constraints, can override)
                 -> PPO policy (learned action selection)
                 -> environment
                 -> reward -> PPO buffer + episodic/semantic memory update

The symbolic safety layer runs LAST, after the PPO agent proposes an
action - it can veto/override the proposed action, but never the other
way around. This mirrors safety/constitutional_ai.py's role: hard
rules are a filter over a learned policy, not something the policy can
learn its way around.
"""
from __future__ import annotations

import numpy as np

from utils.config import NexusConfig
from brain.perception.snn_layer import LIFLayer
from brain.world_model.world_model import WorldModel
from brain.reasoning.symbolic_engine import SymbolicEngine, Rule
from brain.reasoning.planner import MCTSPlanner
from brain.rl.ppo_agent import PPOAgent
from brain.rl.replay_buffer import RolloutBuffer
from brain.memory.working_memory import WorkingMemory
from brain.memory.episodic_memory import EpisodicMemory
from brain.memory.semantic_memory import SemanticMemory


class NexusBrain:
    def __init__(self, sensor_dim: int, state_dim: int, n_actions: int,
                 config: NexusConfig | None = None, seed_stream: str = "nexus_brain"):
        self.config = config or NexusConfig()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.perception = LIFLayer(n_inputs=sensor_dim, config=self.config.snn,
                                    seed_stream=f"{seed_stream}_snn")
        self.world_model = WorldModel(state_dim=state_dim, action_dim=1, config=self.config.mamba,
                                       seed_stream=f"{seed_stream}_wm")
        self.symbolic = SymbolicEngine()
        self.policy = PPOAgent(state_dim=state_dim, n_actions=n_actions, config=self.config.ppo,
                                seed_stream=f"{seed_stream}_ppo")
        self.buffer = RolloutBuffer()

        self.working_memory = WorkingMemory(capacity=64)
        self.episodic_memory = EpisodicMemory(max_episodes=500)
        self.semantic_memory = SemanticMemory()

        self._register_default_safety_rules()
        self._episode_states: list[np.ndarray] = []
        self._episode_actions: list = []
        self._episode_rewards: list = []

    def _register_default_safety_rules(self) -> None:
        """A minimal example rule set - real deployments should load
        domain-specific rules via self.symbolic.add_rule(...) before
        running. This exists so NexusBrain is runnable out of the box."""
        self.symbolic.add_rule(Rule(
            name="veto_on_critical_battery",
            condition=lambda f: f.get("battery_pct", 100) < 5,
            action=lambda f: {"veto_action": True, "override_action": 0},
            priority=100,
        ))

    def perceive(self, sensor_reading_sequence: np.ndarray) -> np.ndarray:
        """sensor_reading_sequence: (T, sensor_dim) -> feature vector (n_snn_neurons,)
        via mean spike rate. In practice T is a short recent window of
        raw sensor samples (e.g. a few ms of lidar/IMU ticks)."""
        return self.perception.spike_rate(sensor_reading_sequence)

    def decide(self, state: np.ndarray, world_facts: dict) -> tuple[int, float, float, bool]:
        """Returns (action, log_prob, value, was_vetoed)."""
        action, log_prob, value = self.policy.act(state)

        self.symbolic.set_facts({**world_facts, "proposed_action": action})
        self.symbolic.forward_chain()

        was_vetoed = bool(self.symbolic.facts.get("veto_action", False))
        if was_vetoed:
            action = int(self.symbolic.facts.get("override_action", action))
        return action, log_prob, value, was_vetoed

    def record_transition(self, state, action, log_prob, reward, value, done) -> None:
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.working_memory.add(state, action, reward)
        self._episode_states.append(state)
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)

        if done:
            self.episodic_memory.store(
                np.array(self._episode_states), self._episode_actions, np.array(self._episode_rewards)
            )
            self._episode_states, self._episode_actions, self._episode_rewards = [], [], []

    def learn(self, last_value: float = 0.0) -> dict:
        """Runs a PPO update over whatever's accumulated in the buffer,
        then clears it (PPO is on-policy)."""
        stats = self.policy.update(self.buffer, last_value=last_value)
        self.buffer.clear()
        return stats