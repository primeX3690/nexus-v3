"""
core/genetic_engine.py

The main evolution loop: seed a population with an initial (possibly
suboptimal) implementation, repeatedly mutate + select by fitness across
generations, track the best individual found.

This is genuine genetic programming over executable Python, not a
metaphor - fitness is measured by actually running each candidate
against test cases (see fitness_scorer.py), and structure is changed
by actually mutating the AST (see ast_mutator.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.config import GeneticConfig
from utils.logger import get_logger
from utils.seeding import get_rng
from core.ast_mutator import ASTMutator
from core.fitness_scorer import evaluate_fitness
from core.population import Population, Individual

logger = get_logger(__name__)


@dataclass
class EvolutionResult:
    best_individual: Individual
    generations_run: int
    history: list[dict] = field(default_factory=list)  # per-gen: {gen, best, mean, diversity}


class GeneticEngine:
    def __init__(
        self,
        seed_source: str,
        func_name: str,
        test_cases: list[tuple[tuple, object]],
        edge_case_args: list[tuple] | None = None,
        config: GeneticConfig | None = None,
        seed_stream: str = "genetic_engine",
    ):
        self.func_name = func_name
        self.test_cases = test_cases
        self.edge_case_args = edge_case_args or []
        self.config = config or GeneticConfig()
        self.mutator = ASTMutator(seed_stream=f"{seed_stream}_mutator")
        self.rng = get_rng(seed_stream)
        self.population = Population(
            seed_source, size=self.config.population_size, seed_stream=f"{seed_stream}_pop"
        )

    def _fitness_fn(self, ind: Individual):
        return evaluate_fitness(ind.source, self.func_name, self.test_cases, self.edge_case_args)

    def _make_child(self, parent: Individual, generation: int) -> Individual:
        mutated_source, mutation_name = self.mutator.mutate_source(parent.source)
        child = Individual(
            source=mutated_source,
            generation_born=generation,
            mutation_history=parent.mutation_history + [mutation_name],
        )
        return child

    def run(self, max_generations: int | None = None, target_fitness: float = 0.999,
             verbose: bool = True) -> EvolutionResult:
        max_gen = max_generations or self.config.max_generations
        self.population.evaluate_all(self._fitness_fn)

        history = []
        for gen in range(max_gen):
            best = self.population.best()
            mean = self.population.mean_score()
            div = self.population.diversity()
            history.append({"gen": gen, "best": best.score, "mean": mean, "diversity": div})
            if verbose:
                logger.info(
                    f"gen {gen:3d} | best={best.score:.4f} mean={mean:.4f} diversity={div:.2f}"
                )

            if best.score >= target_fitness:
                if verbose:
                    logger.info(f"target fitness reached at generation {gen}")
                break

            # Elitism: carry the top fraction forward unchanged.
            next_gen = list(self.population.elites(self.config.elite_fraction))

            # Fill the rest via tournament-selected parents + mutation.
            while len(next_gen) < self.population.size:
                parent = self.population.tournament_select(self.config.tournament_size)
                if self.rng.random() < self.config.mutation_rate * 5:  # scale: mutation applied per-child
                    child = self._make_child(parent, gen + 1)
                else:
                    child = Individual(source=parent.source, generation_born=gen + 1,
                                        mutation_history=parent.mutation_history)
                next_gen.append(child)

            self.population.replace_generation(next_gen)
            self.population.evaluate_all(self._fitness_fn)

        final_best = self.population.best()
        return EvolutionResult(best_individual=final_best, generations_run=len(history), history=history)