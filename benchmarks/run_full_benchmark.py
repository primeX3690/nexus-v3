"""
benchmarks/run_full_benchmark.py

Runs every major subsystem end-to-end and saves REAL evidence -
plots + a numeric report - to results/. This is deliberately separate
from tests/ (which checks correctness) and demo/ (which is for live
presentation): this script exists to produce artifacts you can screenshot
or attach directly to a fellowship application, generated from an actual
run on this exact codebase, not hand-written claims.

Run: python3 -m benchmarks.run_full_benchmark
Output: results/*.png, results/RESULTS.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless - no display needed on a server/CI/laptop without a GUI session
import matplotlib.pyplot as plt

from utils.seeding import set_global_seed
from utils.config import GeneticConfig, PPOConfig, SwarmConfig

RESULTS_DIR = Path("results")


def bench_genetic_evolution() -> dict:
    from core.genetic_engine import GeneticEngine

    set_global_seed(42)
    seed_source = "def solve(a, b):\n    return a - b\n"
    tests = [((1, 2), 3), ((5, 5), 10), ((-1, 1), 0), ((10, 20), 30), ((100, -50), 50)]
    edge = [(0, 0), (-1000, 1000), (999999, 1)]

    cfg = GeneticConfig(population_size=25, mutation_rate=0.15, elite_fraction=0.2,
                         tournament_size=3, max_generations=50)
    engine = GeneticEngine(seed_source, "solve", tests, edge, config=cfg)
    t0 = time.perf_counter()
    result = engine.run(verbose=False)
    elapsed = time.perf_counter() - t0

    gens = [h["gen"] for h in result.history]
    best = [h["best"] for h in result.history]
    mean = [h["mean"] for h in result.history]

    plt.figure(figsize=(7, 4))
    plt.plot(gens, best, label="best fitness", linewidth=2)
    plt.plot(gens, mean, label="population mean fitness", alpha=0.6)
    plt.xlabel("generation")
    plt.ylabel("fitness score")
    plt.title("Genetic Code Evolution: fitness over generations\n(seed program deliberately WRONG - evolves a-b into a+b)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "genetic_evolution_fitness.png", dpi=120)
    plt.close()

    return {
        "wall_clock_seconds": round(elapsed, 3),
        "generations_run": result.generations_run,
        "final_best_fitness": round(result.best_individual.score, 4),
        "final_correctness": round(result.best_individual.fitness.correctness, 4),
        "evolved_source": result.best_individual.source,
        "mutation_lineage": result.best_individual.mutation_history,
    }


def bench_ppo_gradient_check() -> dict:
    """Re-runs the finite-difference gradient check and reports the
    actual numeric error - this is the single most important number
    for establishing the hand-rolled backprop is correct, not just
    'ran without crashing'."""
    from brain.rl.ppo_agent import MLP
    from utils.seeding import get_rng

    rng = get_rng("bench_grad_check")
    mlp = MLP(in_dim=4, hidden_dim=6, out_dim=3, rng=rng)
    x = rng.normal(0, 1, (5, 4))
    out = mlp.forward(x)
    d_out = rng.normal(0, 1, out.shape)
    grads = mlp.backward(d_out)

    def loss_fn():
        z1 = x @ mlp.W1 + mlp.b1
        h1 = np.tanh(z1)
        z2 = h1 @ mlp.W2 + mlp.b2
        return np.sum(z2 * d_out)

    eps = 1e-5
    max_rel_errors = {}
    for pname, param in [("W1", mlp.W1), ("b1", mlp.b1), ("W2", mlp.W2), ("b2", mlp.b2)]:
        numerical = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            lp = loss_fn()
            param[idx] = orig - eps
            lm = loss_fn()
            param[idx] = orig
            numerical[idx] = (lp - lm) / (2 * eps)
        analytical = grads[pname]
        rel_err = np.max(np.abs(numerical - analytical)) / (np.max(np.abs(numerical)) + 1e-8)
        max_rel_errors[pname] = float(rel_err)

    return {"max_relative_gradient_error": max_rel_errors,
            "verdict": "PASS (all < 1e-4)" if all(v < 1e-4 for v in max_rel_errors.values()) else "FAIL"}


def bench_ppo_training() -> dict:
    from environment.gym_wrapper import NavigationEnv
    from training.rl_trainer import RLTrainer

    set_global_seed(0)
    env = NavigationEnv(world_size=15, n_obstacles=2, max_steps=60, seed=0)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cfg = PPOConfig(lr=3e-4, epochs_per_update=4, minibatch_size=32, entropy_coef=0.02)
    trainer = RLTrainer(env, state_dim=obs_dim, n_actions=n_actions, config=cfg, use_safety_layer=True)
    t0 = time.perf_counter()
    history = trainer.train(n_updates=25, steps_per_update=128, verbose=False)
    elapsed = time.perf_counter() - t0

    rewards = [h["mean_episode_reward"] for h in history if h["mean_episode_reward"] is not None]
    updates = list(range(len(rewards)))

    plt.figure(figsize=(7, 4))
    plt.plot(updates, rewards, marker="o", markersize=3)
    if len(rewards) > 5:
        window = 5
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), smoothed, linewidth=2, label=f"{window}-update moving avg")
        plt.legend()
    plt.xlabel("PPO update")
    plt.ylabel("mean episode reward")
    plt.title("PPO Training: obstacle-avoidance navigation task\n(hand-rolled MLP + manual backprop, gradient-checked)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ppo_training_reward.png", dpi=120)
    plt.close()

    return {
        "wall_clock_seconds": round(elapsed, 3),
        "n_updates": len(history),
        "total_env_steps": len(history) * 128,
        "first_5_updates_mean_reward": round(float(np.mean(rewards[:5])), 3) if len(rewards) >= 5 else None,
        "last_5_updates_mean_reward": round(float(np.mean(rewards[-5:])), 3) if len(rewards) >= 5 else None,
    }


def bench_swarm_fault_tolerance() -> dict:
    from simulation.swarm.swarm_manager import SwarmManager

    set_global_seed(7)
    sm = SwarmManager(config=SwarmConfig(n_agents=6, comm_range=15, heartbeat_interval_steps=4,
                                          heartbeat_timeout_steps=12), world_size=(30, 30))

    alive_counts, spreads, steps = [], [], []
    detection_step = None
    t0 = time.perf_counter()
    for t in range(80):
        if t == 20:
            sm.kill_agent(2)
            sm.kill_agent(4)
        info = sm.step()
        alive_counts.append(info["alive_count"])
        spreads.append(info["spread"])
        steps.append(t)
        if info["newly_dead_detections"] and detection_step is None:
            detection_step = t
    elapsed = time.perf_counter() - t0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, alive_counts, drawstyle="steps-post")
    ax1.axvline(20, color="red", linestyle="--", alpha=0.5, label="2 agents killed")
    ax1.set_xlabel("step")
    ax1.set_ylabel("alive agent count")
    ax1.set_title("Decentralized fault detection\n(no central coordinator)")
    ax1.legend()

    ax2.plot(steps, spreads)
    ax2.set_xlabel("step")
    ax2.set_ylabel("swarm spread (mean dist from centroid)")
    ax2.set_title("Swarm cohesion over time\n(boids + stigmergy, bounded world)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "swarm_fault_tolerance.png", dpi=120)
    plt.close()

    return {
        "wall_clock_seconds": round(elapsed, 3),
        "steps_simulated": 80,
        "agents_killed_at_step": 20,
        "failure_detected_at_step": detection_step,
        "detection_latency_steps": (detection_step - 20) if detection_step else None,
        "final_alive_count": alive_counts[-1],
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Running full benchmark suite (laptop-only, no hardware needed)...")
    print()

    report = {}

    print("[1/4] Genetic code evolution...")
    report["genetic_evolution"] = bench_genetic_evolution()

    print("[2/4] PPO backprop gradient check (finite differences)...")
    report["ppo_gradient_check"] = bench_ppo_gradient_check()

    print("[3/4] PPO training on navigation task...")
    report["ppo_training"] = bench_ppo_training()

    print("[4/4] Swarm decentralized fault tolerance...")
    report["swarm_fault_tolerance"] = bench_swarm_fault_tolerance()

    (RESULTS_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2, default=str))

    md = ["# NEXUS v3 - Benchmark Results\n",
          "Generated by `benchmarks/run_full_benchmark.py`. Every number below comes from an "
          "actual run of this exact code, on this exact commit - not a hand-written claim.\n"]

    md.append("## 1. Genetic Code Evolution\n")
    ge = report["genetic_evolution"]
    md.append(f"- Ran {ge['generations_run']} generations in {ge['wall_clock_seconds']}s\n")
    md.append(f"- Final fitness: **{ge['final_best_fitness']}** (correctness: {ge['final_correctness']})\n")
    md.append(f"- Evolved via mutation(s): `{ge['mutation_lineage']}`\n")
    md.append(f"- Evolved source:\n```python\n{ge['evolved_source']}```\n")
    md.append("![genetic evolution](genetic_evolution_fitness.png)\n")

    md.append("## 2. PPO Backprop Correctness (Finite-Difference Gradient Check)\n")
    gc = report["ppo_gradient_check"]
    md.append(f"- Verdict: **{gc['verdict']}**\n")
    md.append(f"- Max relative error per parameter: `{gc['max_relative_gradient_error']}`\n")

    md.append("## 3. PPO Training (Obstacle-Avoidance Navigation)\n")
    pt = report["ppo_training"]
    md.append(f"- {pt['total_env_steps']} environment steps in {pt['wall_clock_seconds']}s "
               f"({round(pt['total_env_steps']/pt['wall_clock_seconds'])} steps/sec, CPU-only)\n")
    md.append(f"- Mean reward: first 5 updates = {pt['first_5_updates_mean_reward']}, "
               f"last 5 updates = {pt['last_5_updates_mean_reward']}\n")
    md.append("![ppo training](ppo_training_reward.png)\n")

    md.append("## 4. Swarm Decentralized Fault Tolerance\n")
    sw = report["swarm_fault_tolerance"]
    md.append(f"- 2 of 6 agents killed at step {sw['agents_killed_at_step']}, "
               f"detected by peers at step {sw['failure_detected_at_step']} "
               f"(latency: {sw['detection_latency_steps']} steps) - **no central coordinator involved**\n")
    md.append(f"- Final alive count: {sw['final_alive_count']}/6\n")
    md.append("![swarm fault tolerance](swarm_fault_tolerance.png)\n")

    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md))

    print()
    print(f"Done. Wrote plots + RESULTS.md + benchmark_report.json to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()