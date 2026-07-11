"""
demo/run_multi_robot.py

The flagship demo: a 5-robot swarm that
  1. flocks and searches cooperatively via stigmergy (simulation/swarm/)
  2. tolerates simulated robot failures with decentralized heartbeat
     detection (no central coordinator - killing the "leader" doesn't
     break anything, because there isn't one)
  3. each robot learns its own steering policy via independent PPO
     (training/multi_agent_trainer.py)

Run: python -m demo.run_multi_robot
"""
from __future__ import annotations

import numpy as np

from utils.seeding import set_global_seed
from utils.config import SwarmConfig, PPOConfig
from environment.multi_robot_env import MultiRobotEnv
from training.multi_agent_trainer import MultiAgentTrainer


def main():
    set_global_seed(7)

    print("=" * 70)
    print("NEXUS v3 - 5-Robot Swarm Civilization Demo")
    print("=" * 70)

    world_size = (30, 30)
    target = np.array([25.0, 25.0])
    swarm_cfg = SwarmConfig(n_agents=5, comm_range=12, heartbeat_interval_steps=4,
                             heartbeat_timeout_steps=12)
    env = MultiRobotEnv(config=swarm_cfg, world_size=world_size, target_pos=target, max_steps=60)
    agent_ids = [a.agent_id for a in env.swarm.agents]

    ppo_cfg = PPOConfig(lr=3e-4, epochs_per_update=3, minibatch_size=16, entropy_coef=0.02)
    trainer = MultiAgentTrainer(env, agent_ids=agent_ids, state_dim=6, n_actions=5, config=ppo_cfg)

    print(f"Swarm: {swarm_cfg.n_agents} agents, world={world_size}, target={target}")
    print(f"Each agent trains an INDEPENDENT policy - decentralized, no central coordinator.")
    print()

    n_rounds = 10
    for round_i in range(n_rounds):
        history = trainer.train(n_updates=1, steps_per_update=60, verbose=False)
        record = history[0]

        # Simulate a robot failure partway through, to demonstrate
        # fault-tolerant continuation.
        if round_i == 4:
            print(f">>> round {round_i}: simulating failure of agent 1 <<<")
            env.kill_agent(1)

        rewards = [record.get(f"agent{aid}_mean_reward") for aid in agent_ids]
        alive = sum(1 for a in env.swarm.agents if a.alive)
        print(f"round {round_i:2d} | alive={alive}/5 | rewards={['%.2f' % r if r else None for r in rewards]}")

    print()
    print("Final swarm state:")
    for a in env.swarm.agents:
        status = "ALIVE" if a.alive else "DEAD"
        print(f"  agent {a.agent_id}: {status}, position={a.position.round(2)}")
    print(f"Final centroid: {env.swarm.centroid().round(2)}  (target: {target})")
    print("=" * 70)


if __name__ == "__main__":
    main()