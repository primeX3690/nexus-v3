"""
benchmarks/profile_resources.py

Real memory + CPU profiling of every major NEXUS subsystem, using
psutil (process-level RSS/CPU) and tracemalloc (Python allocation
tracking). Every number in results/RESOURCE_PROFILE.md comes from
actually running the code and measuring it - not estimated.

Run: python3 -m benchmarks.profile_resources
"""
from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

RESULTS_DIR = Path("results")
PROCESS = psutil.Process()


def _measure(label: str, fn, *args, **kwargs) -> dict:
    """Runs fn once, measuring: wall-clock time, peak Python-allocation
    memory (tracemalloc), and process RSS delta (psutil) - two
    different memory lenses because tracemalloc only sees Python-level
    allocations (misses numpy's C-level buffers), while RSS captures
    everything the OS actually mapped for this process, including numpy."""
    gc.collect()
    rss_before = PROCESS.memory_info().rss / (1024 ** 2)  # MB
    cpu_before = PROCESS.cpu_times()

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_after = PROCESS.cpu_times()
    rss_after = PROCESS.memory_info().rss / (1024 ** 2)

    return {
        "label": label,
        "wall_clock_seconds": round(elapsed, 4),
        "tracemalloc_peak_mb": round(peak / (1024 ** 2), 3),
        "process_rss_before_mb": round(rss_before, 2),
        "process_rss_after_mb": round(rss_after, 2),
        "process_rss_delta_mb": round(rss_after - rss_before, 2),
        "cpu_user_seconds": round(cpu_after.user - cpu_before.user, 4),
        "cpu_system_seconds": round(cpu_after.system - cpu_before.system, 4),
        "result_summary": result,
    }


def profile_mamba_ssm():
    from brain.world_model.mamba_ssm import MambaSSM
    from utils.config import MambaConfig
    from utils.seeding import set_global_seed

    def run():
        set_global_seed(0)
        ssm = MambaSSM(MambaConfig(d_model=64, d_state=16))
        x = np.random.default_rng(1).normal(0, 1, (2000, 64))
        y = ssm.run_sequence(x)
        return {"sequence_length": 2000, "d_model": 64, "output_shape": list(y.shape)}

    return _measure("Mamba SSM (2000-step sequential scan, d_model=64)", run)


def profile_ppo_training():
    from environment.gym_wrapper import NavigationEnv
    from training.rl_trainer import RLTrainer
    from utils.config import PPOConfig
    from utils.seeding import set_global_seed

    def run():
        set_global_seed(0)
        env = NavigationEnv(world_size=15, n_obstacles=2, max_steps=60, seed=0)
        cfg = PPOConfig(lr=3e-4, epochs_per_update=4, minibatch_size=32)
        trainer = RLTrainer(env, state_dim=env.observation_space.shape[0],
                             n_actions=env.action_space.n, config=cfg, use_safety_layer=True)
        history = trainer.train(n_updates=15, steps_per_update=128, verbose=False)
        return {"n_updates": len(history), "total_env_steps": len(history) * 128}

    return _measure("PPO training (hand-rolled MLP+backprop, 15 updates x 128 steps)", run)


def profile_swarm_5_robots():
    from simulation.swarm.swarm_manager import SwarmManager
    from utils.config import SwarmConfig
    from utils.seeding import set_global_seed

    def run():
        set_global_seed(0)
        sm = SwarmManager(config=SwarmConfig(n_agents=5), world_size=(30, 30))
        for _ in range(200):
            sm.step()
        return {"n_agents": 5, "steps_simulated": 200}

    return _measure("5-robot swarm (200 steps: boids + stigmergy + fault tolerance)", run)


def profile_genetic_evolution():
    from core.genetic_engine import GeneticEngine
    from utils.config import GeneticConfig
    from utils.seeding import set_global_seed

    def run():
        set_global_seed(42)
        seed_source = "def solve(a, b):\n    return a - b\n"
        tests = [((1, 2), 3), ((5, 5), 10), ((-1, 1), 0)]
        cfg = GeneticConfig(population_size=25, max_generations=50)
        engine = GeneticEngine(seed_source, "solve", tests, config=cfg)
        result = engine.run(verbose=False)
        return {"generations_run": result.generations_run, "final_fitness": round(result.best_individual.score, 4)}

    return _measure("Genetic evolution (population=25, up to 50 generations)", run)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"System: {psutil.cpu_count(logical=False)} physical cores, "
          f"{psutil.cpu_count(logical=True)} logical, "
          f"{psutil.virtual_memory().total / (1024**3):.1f} GB total RAM")
    print()

    profiles = []
    for name, fn in [("mamba_ssm", profile_mamba_ssm), ("ppo_training", profile_ppo_training),
                      ("swarm_5_robots", profile_swarm_5_robots), ("genetic_evolution", profile_genetic_evolution)]:
        print(f"Profiling {name}...")
        p = fn()
        profiles.append(p)
        print(f"  wall_clock={p['wall_clock_seconds']}s  "
              f"tracemalloc_peak={p['tracemalloc_peak_mb']}MB  "
              f"rss_delta={p['process_rss_delta_mb']}MB  "
              f"cpu_user={p['cpu_user_seconds']}s")

    system_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    report = {"system_info": system_info, "profiles": profiles}
    (RESULTS_DIR / "resource_profile_report.json").write_text(json.dumps(report, indent=2, default=str))

    md = ["# NEXUS v3 - Resource Profile\n",
          f"System this was measured on: {system_info['physical_cores']} physical cores, "
          f"{system_info['logical_cores']} logical, {system_info['total_ram_gb']} GB RAM.\n",
          "Every number below is from an actual `psutil`/`tracemalloc` measurement of this "
          "exact code running - not an estimate.\n",
          "| Subsystem | Wall clock | Peak Python allocations (tracemalloc) | "
          "Process RSS delta | CPU (user+sys) |",
          "|---|---|---|---|---|"]
    for p in profiles:
        md.append(f"| {p['label']} | {p['wall_clock_seconds']}s | {p['tracemalloc_peak_mb']} MB | "
                   f"{p['process_rss_delta_mb']} MB | {p['cpu_user_seconds'] + p['cpu_system_seconds']:.3f}s |")
    md.append("\n**Note on RSS delta:** this measures memory added to the process *during* "
               "that specific subsystem's run - it does not include the baseline Python/numpy "
               "interpreter footprint (typically 30-60MB) that exists before any of this code runs.\n")
    (RESULTS_DIR / "RESOURCE_PROFILE.md").write_text("\n".join(md))

    print(f"\nWrote results/RESOURCE_PROFILE.md, results/resource_profile_report.json")


if __name__ == "__main__":
    main()