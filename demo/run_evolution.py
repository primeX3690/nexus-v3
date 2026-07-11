"""
demo/run_evolution.py

Live demo of the genetic programming engine: seeds a deliberately wrong
implementation of a target function and evolves it toward correctness
purely through AST mutation + execution-based fitness scoring. No LLM
involved anywhere in this loop.

Run: python -m demo.run_evolution
"""
from __future__ import annotations

from utils.seeding import set_global_seed
from utils.config import GeneticConfig
from core.genetic_engine import GeneticEngine


def main():
    set_global_seed(42)

    print("=" * 70)
    print("NEXUS v3 - Live Genetic Code Evolution Demo")
    print("=" * 70)
    print()
    print("Seed program (deliberately WRONG - computes a-b instead of a+b):")
    seed_source = "def solve(a, b):\n    return a - b\n"
    print(seed_source)

    test_cases = [((1, 2), 3), ((5, 5), 10), ((-1, 1), 0), ((10, 20), 30), ((100, -50), 50)]
    edge_cases = [(0, 0), (-1000, 1000), (999999, 1)]

    cfg = GeneticConfig(population_size=25, mutation_rate=0.15, elite_fraction=0.2,
                         tournament_size=3, max_generations=50)
    engine = GeneticEngine(seed_source, "solve", test_cases, edge_cases, config=cfg)

    print(f"Evolving over up to {cfg.max_generations} generations, population={cfg.population_size}...")
    print()
    result = engine.run(verbose=True)

    print()
    print("=" * 70)
    print(f"Evolution finished after {result.generations_run} generations")
    print(f"Best fitness score: {result.best_individual.score:.4f}")
    print(f"Mutation lineage: {result.best_individual.mutation_history}")
    print()
    print("Evolved source code:")
    print(result.best_individual.source)
    print("=" * 70)


if __name__ == "__main__":
    main()