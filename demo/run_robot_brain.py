"""
demo/run_robot_brain.py

Single-robot training demo: wires NavigationEnv (physics-lite gym env)
with NexusBrain's full stack - SNN perception, PPO policy, symbolic
safety veto, working/episodic memory - and trains for a modest number
of PPO updates, printing progress.

Run: python -m demo.run_robot_brain
"""
from __future__ import annotations

import numpy as np

from utils.seeding import set_global_seed
from utils.config import NexusConfig
from environment.gym_wrapper import NavigationEnv
from brain.nexus_brain import NexusBrain


def main():
    set_global_seed(0)

    print("=" * 70)
    print("NEXUS v3 - Single Robot Brain Training Demo")
    print("(SNN perception -> PPO policy -> symbolic safety veto -> memory)")
    print("=" * 70)

    env = NavigationEnv(world_size=15, n_obstacles=3, max_steps=80, seed=0)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    config = NexusConfig()
    config.ppo.lr = 3e-4
    config.ppo.entropy_coef = 0.02

    brain = NexusBrain(sensor_dim=obs_dim, state_dim=obs_dim, n_actions=n_actions, config=config)

    n_updates = 20
    steps_per_update = 160
    obs, _ = env.reset()
    total_vetoes = 0

    for update_i in range(n_updates):
        ep_rewards, ep_sum = [], 0.0
        for _ in range(steps_per_update):
            state = obs  # using raw obs directly as 'state' for this demo (skipping SNN encoding
                         # of a single vector - SNN perception is meant for raw sensor STREAMS,
                         # demonstrated separately; see tests/test_brain_core.py)
            world_facts = {
                "battery_pct": 100,  # not modeled in this simple env; always healthy
                "in_bounds": True,
                "min_obstacle_distance": min(
                    (np.linalg.norm(env.robot_pos - o) for o in env.obstacles), default=999
                ),
            }
            action, log_prob, value, vetoed = brain.decide(state, world_facts)
            total_vetoes += int(vetoed)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            brain.record_transition(state, action, log_prob, reward, value, done)

            ep_sum += reward
            if done:
                ep_rewards.append(ep_sum)
                ep_sum = 0.0
                next_obs, _ = env.reset()
            obs = next_obs

        stats = brain.learn()
        mean_r = np.mean(ep_rewards) if ep_rewards else float("nan")
        print(f"update {update_i:3d} | episodes={len(ep_rewards):2d} mean_reward={mean_r:7.3f} "
              f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f}")

    print()
    print(f"Total safety vetoes triggered during training: {total_vetoes}")
    print(f"Episodic memory holds {len(brain.episodic_memory)} stored episodes")
    print(f"Best remembered episode total reward: {brain.episodic_memory.best_episodes(1)[0].total_reward:.3f}")

    checkpoint_path = "checkpoints/robot_brain_ppo"
    import os
    os.makedirs("checkpoints", exist_ok=True)
    brain.policy.save(checkpoint_path)
    print(f"Trained policy saved to {checkpoint_path}.npz "
          f"(reload with agent.load('{checkpoint_path}') to demo without retraining)")
    print("=" * 70)


if __name__ == "__main__":
    main()