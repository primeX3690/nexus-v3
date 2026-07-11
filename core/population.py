"""
core/population.py

Manages a population of candidate programs (genes = source strings),
their fitness scores, elitism, tournament selection, and basic
diversity tracking (to detect premature convergence, a very common
genetic-programming failure mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.seeding import get_rng
from core.fitness_scorer import FitnessScore


@dataclass
class Individual:
    source: str
    fitness: FitnessScore | None = None
    generation_born: int = 0
    mutation_history: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.fitness.total if self.fitness is not None else -1.0


class Population:
    def __init__(self, seed_source: str, size: int = 30, seed_stream: str = "population"):
        self.size = size
        self.rng = get_rng(seed_stream)
        self.generation = 0
        self.individuals: list[Individual] = [
            Individual(source=seed_source, generation_born=0) for _ in range(size)
        ]

    def evaluate_all(self, fitness_fn) -> None:
        """fitness_fn: Individual -> FitnessScore"""
        for ind in self.individuals:
            if ind.fitness is None:
                ind.fitness = fitness_fn(ind)

    def best(self) -> Individual:
        return max(self.individuals, key=lambda i: i.score)

    def mean_score(self) -> float:
        scores = [i.score for i in self.individuals]
        return sum(scores) / len(scores) if scores else 0.0

    def diversity(self) -> float:
        """Fraction of unique source strings in the population. 1.0 =
        fully diverse, near 0 = converged (possibly prematurely)."""
        unique = len(set(i.source for i in self.individuals))
        return unique / len(self.individuals) if self.individuals else 0.0

    def tournament_select(self, k: int = 3) -> Individual:
        contenders_idx = self.rng.choice(len(self.individuals), size=min(k, len(self.individuals)), replace=False)
        contenders = [self.individuals[i] for i in contenders_idx]
        return max(contenders, key=lambda i: i.score)

    def elites(self, fraction: float = 0.2) -> list[Individual]:
        n_elite = max(1, int(len(self.individuals) * fraction))
        ranked = sorted(self.individuals, key=lambda i: i.score, reverse=True)
        return ranked[:n_elite]

    def replace_generation(self, new_individuals: list[Individual]) -> None:
        assert len(new_individuals) == self.size, \
            f"population size must stay constant: got {len(new_individuals)}, expected {self.size}"
        self.individuals = new_individuals
        self.generation += 1