"""
demo/run_co_evolution.py

Live demo of morphological co-evolution: code AND physical hardware
parameters (sensor range, max speed, comm range) evolve together
against a task + hardware-cost fitness function. See
core/co_evolution.py's module docstring for an honest account of a real
"bloat" issue found while testing this (duplicate function definitions
becoming dead code) and how the fitness evaluation was fixed in
response.

Run: python -m demo.run_co_evolution
"""
from __future__ import annotations
from utils.seeding import set_global_seed
from core.co_evolution import CoEvolutionEngine


def main():
    set_global_seed(1)

    print("=" * 70)
    print("NEXUS v3 - Morphological Co-Evolution Demo")
    print("(control code AND physical hardware parameters evolve together)")
    print("=" * 70)
    print()
    print("Task: reach a target 20 units away, avoid an obstacle in between.")
    print("Fitness = task success - 0.05 * hardware_cost(sensor_range, max_speed, comm_range)")
    print("Hardware cost is super-linear in sensor_range - bigger sensors cost more,")
    print("so evolution is pressured to find the CHEAPEST hardware that still works,")
    print("not just maximize every physical parameter.")
    print()

    engine = CoEvolutionEngine(population_size=25)
    initial_mean_sensor = sum(i.morphology.sensor_range for i in engine.population) / len(engine.population)
    print(f"Initial population mean sensor_range: {initial_mean_sensor:.2f}")
    print()

    history = engine.run(generations=40, verbose=True)

    best = sorted(engine.population, key=lambda i: i.fitness, reverse=True)[0]
    print()
    print("=" * 70)
    print(f"Sensor range: {initial_mean_sensor:.2f} (initial mean) -> "
          f"{best.morphology.sensor_range:.2f} (best evolved)")
    print(f"Final: task_score={best.task_score:.3f}  hardware_cost={best.hardware_cost:.3f}  "
          f"fitness={best.fitness:.3f}")
    print(f"Best morphology: {best.morphology}")
    print()
    print("Best evolved decision code:")
    print(best.source)
    print("=" * 70)


if __name__ == "__main__":
    main()