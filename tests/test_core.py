import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
import pytest
from utils.seeding import set_global_seed, get_rng
from core.ast_mutator import ASTMutator, MUTATION_OPERATORS
from core.fitness_scorer import evaluate_fitness
from core.population import Population, Individual
from core.genetic_engine import GeneticEngine
from utils.config import GeneticConfig


def test_all_12_mutation_operators_registered():
    assert len(MUTATION_OPERATORS) == 12


def test_mutator_always_produces_valid_syntax():
    set_global_seed(5)
    src = "def f(x):\n    total = 0\n    for i in range(10):\n        if x > 5:\n            total = total + i\n        else:\n            total = total - i\n    return total\n"
    m = ASTMutator()
    for _ in range(100):
        out, name = m.mutate_source(src)
        compile(out, "<t>", "exec")  # raises SyntaxError if invalid


def test_fitness_scorer_correct_vs_incorrect():
    tests = [((1, 2), 3), ((5, 5), 10)]
    good = evaluate_fitness("def add(a,b):\n    return a+b\n", "add", tests)
    bad = evaluate_fitness("def add(a,b):\n    return a-b\n", "add", tests)
    assert good.correctness == 1.0
    assert bad.correctness == 0.0
    assert good.total > bad.total


def test_population_elitism_and_selection():
    set_global_seed(1)
    pop = Population("def f(): pass", size=10)
    for i, ind in enumerate(pop.individuals):
        from core.fitness_scorer import FitnessScore
        ind.fitness = FitnessScore(correctness=i / 10)
    elites = pop.elites(fraction=0.2)
    assert len(elites) == 2
    # score is the weighted total (correctness weight = 0.45), not raw correctness
    assert all(e.fitness.correctness >= 0.7 for e in elites)  # top 2 of 0..0.9


def test_genetic_engine_improves_fitness_over_generations():
    set_global_seed(42)
    seed_source = "def solve(a, b):\n    return a - b\n"
    tests = [((1, 2), 3), ((5, 5), 10), ((-1, 1), 0)]
    cfg = GeneticConfig(population_size=15, max_generations=30)
    engine = GeneticEngine(seed_source, "solve", tests, config=cfg)
    result = engine.run(verbose=False)
    assert result.best_individual.fitness.correctness == 1.0
    assert result.history[0]["best"] <= result.best_individual.score


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))