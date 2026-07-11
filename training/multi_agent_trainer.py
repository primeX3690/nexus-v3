"""
training/multi_agent_trainer.py

Decentralized multi-agent training: each robot in the swarm has its OWN
independent PPOAgent and RolloutBuffer, learning purely from its own
local observations/rewards - there is no central critic, no parameter
sharing, and no central coordinator making decisions for the group.
This mirrors the "no single point of failure" philosophy already
enforced in simulation/swarm/fault_tolerance.py: if a training process
for one agent crashed, it wouldn't take down the others' learning.

"Synchronization" here means only this: after each rollout collection
phase, every agent's independent update happens in the same round, so
progress stays comparable/timed together for logging - not that they
share gradients or weights.
"""
from __future__ import annotations

import numpy as np

from brain.rl.ppo_agent import PPOAgent
from brain.rl.replay_buffer import RolloutBuffer
from utils.config import PPOConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiAgentTrainer:
    def __init__(self, env, agent_ids: list, state_dim: int, n_actions: int,
                 config: PPOConfig | None = None, seed_stream: str = "multi_agent_trainer"):
        self.env = env
        self.agent_ids = list(agent_ids)
        self.policies = {
            aid: PPOAgent(state_dim=state_dim, n_actions=n_actions, config=config,
                           seed_stream=f"{seed_stream}_agent{aid}")
            for aid in self.agent_ids
        }
        self.buffers = {aid: RolloutBuffer() for aid in self.agent_ids}

    def collect_rollout(self, n_steps: int, obs: dict) -> tuple[dict, dict]:
        ep_reward_sums = {aid: 0.0 for aid in self.agent_ids}
        ep_reward_history = {aid: [] for aid in self.agent_ids}

        for _ in range(n_steps):
            actions, log_probs, values = {}, {}, {}
            for aid in self.agent_ids:
                if aid not in obs:
                    continue  # agent may have been marked dead by fault tolerance
                a, lp, v = self.policies[aid].act(obs[aid])
                actions[aid], log_probs[aid], values[aid] = a, lp, v

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            for aid in actions:
                self.buffers[aid].add(obs[aid], actions[aid], log_probs[aid],
                                       rewards.get(aid, 0.0), values[aid], done)
                ep_reward_sums[aid] += rewards.get(aid, 0.0)

            if done:
                for aid in self.agent_ids:
                    ep_reward_history[aid].append(ep_reward_sums[aid])
                    ep_reward_sums[aid] = 0.0
                next_obs = self.env.reset()

            obs = next_obs

        return obs, ep_reward_history

    def train(self, n_updates: int, steps_per_update: int = 128, verbose: bool = True) -> list[dict]:
        obs = self.env.reset()
        history = []

        for update_i in range(n_updates):
            obs, ep_rewards = self.collect_rollout(steps_per_update, obs)

            round_stats = {"update": update_i}
            for aid in self.agent_ids:
                if len(self.buffers[aid]) == 0:
                    continue  # this agent got no experience this round (e.g. died)
                last_val = 0.0
                update_stats = self.policies[aid].update(self.buffers[aid], last_value=last_val)
                self.buffers[aid].clear()
                round_stats[f"agent{aid}_policy_loss"] = update_stats["policy_loss"]
                round_stats[f"agent{aid}_mean_reward"] = (
                    float(np.mean(ep_rewards[aid])) if ep_rewards[aid] else None
                )

            history.append(round_stats)
            if verbose:
                rewards_str = ", ".join(
                    f"a{aid}={round_stats.get(f'agent{aid}_mean_reward')}" for aid in self.agent_ids
                )
                logger.info(f"update {update_i:3d} | {rewards_str}")

        return history