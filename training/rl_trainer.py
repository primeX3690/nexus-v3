"""
training/rl_trainer.py

Standard on-policy training loop: collect a fixed number of environment
steps into a RolloutBuffer, run a PPO update, repeat. The safety layer
is applied to every action before it hits the environment - training
happens under the same constraints the robot will face at deployment,
not bolted on afterward.
"""
from __future__ import annotations

import numpy as np

from brain.rl.ppo_agent import PPOAgent
from brain.rl.replay_buffer import RolloutBuffer
from safety.constitutional_ai import ConstitutionalSafetyLayer
from utils.config import PPOConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class RLTrainer:
    def __init__(self, env, state_dim: int, n_actions: int, config: PPOConfig | None = None,
                 use_safety_layer: bool = True, seed_stream: str = "rl_trainer"):
        self.env = env
        self.agent = PPOAgent(state_dim=state_dim, n_actions=n_actions, config=config,
                               seed_stream=seed_stream)
        self.safety = ConstitutionalSafetyLayer() if use_safety_layer else None
        self.buffer = RolloutBuffer()

    def _world_facts_from_env(self, env) -> dict:
        """Best-effort extraction of safety-relevant facts from whatever
        env is plugged in; environments that don't expose these fields
        simply get permissive defaults (no vetoes triggered)."""
        facts = {"in_bounds": True}
        if hasattr(env, "robot_pos") and hasattr(env, "world_size"):
            facts["in_bounds"] = bool(np.all(env.robot_pos >= 0) and np.all(env.robot_pos <= env.world_size))
        if hasattr(env, "obstacles") and hasattr(env, "robot_pos") and len(getattr(env, "obstacles", [])) > 0:
            dists = [np.linalg.norm(env.robot_pos - o) for o in env.obstacles]
            facts["min_obstacle_distance"] = float(min(dists))
        return facts

    def collect_rollout(self, n_steps: int, obs: np.ndarray) -> tuple[np.ndarray, float, dict]:
        """Runs n_steps of env interaction, filling self.buffer. Returns
        (final_obs, last_value_estimate, episode_stats)."""
        ep_rewards, ep_reward_sum = [], 0.0
        for _ in range(n_steps):
            action, log_prob, value = self.agent.act(obs)

            if self.safety is not None:
                facts = self._world_facts_from_env(self.env)
                action, vetoed, _ = self.safety.check(action, facts)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(obs, action, log_prob, reward, value, done)

            ep_reward_sum += reward
            if done:
                ep_rewards.append(ep_reward_sum)
                ep_reward_sum = 0.0
                next_obs, _ = self.env.reset()

            obs = next_obs

        last_value = 0.0 if done else float(self.agent.value_net.forward(obs[None, :])[0, 0])
        stats = {"mean_episode_reward": float(np.mean(ep_rewards)) if ep_rewards else None,
                 "n_episodes": len(ep_rewards)}
        return obs, last_value, stats

    def train(self, n_updates: int, steps_per_update: int = 256, verbose: bool = True) -> list[dict]:
        obs, _ = self.env.reset()
        history = []
        for update_i in range(n_updates):
            obs, last_value, rollout_stats = self.collect_rollout(steps_per_update, obs)
            update_stats = self.agent.update(self.buffer, last_value=last_value)
            self.buffer.clear()

            record = {**rollout_stats, **update_stats, "update": update_i}
            history.append(record)
            if verbose:
                logger.info(
                    f"update {update_i:3d} | mean_ep_reward={rollout_stats['mean_episode_reward']} "
                    f"policy_loss={update_stats['policy_loss']:.4f} value_loss={update_stats['value_loss']:.4f}"
                )
        return history