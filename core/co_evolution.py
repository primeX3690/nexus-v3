"""
core/co_evolution.py

Runs code evolution (core/ast_mutator.py) and morphology evolution
(core/morphology.py) together against ONE fitness function, on a
simple obstacle-avoidance task: given (distance_to_target,
distance_to_obstacle), decide an acceleration. Fitness rewards reaching
the target without hitting the obstacle, MINUS a hardware cost term -
so evolution is genuinely pressured to trade off between "smarter code"
and "better sensors/motors," not just maximize both.

The task is deliberately simple (1D, hand-rolled physics) so the
co-evolution DYNAMIC is the thing being demonstrated and verified, not
the task's realism - see environment/gym_wrapper.py for the more
realistic 2D navigation task used elsewhere in this repo.

HONEST FINDING FROM TESTING THIS: early runs showed a textbook "bloat"
problem from genetic programming literature - the statement_duplicate
mutation operator can duplicate a whole top-level function definition,
and since Python's exec() keeps only the LAST definition of a given
name, earlier (possibly more sophisticated, sensor-using) function
bodies silently become dead code with no effect. Combined with an
initial fitness function using only 5 randomly-sampled obstacle
positions, this let a trivial "ignore the sensor, always move at max
speed" solution score a perfect task_score purely by evaluation luck
(see git history / session notes for the specific numbers). Fixed here
by evaluating against a FIXED grid of obstacle positions rather than
random sampling - but the bloat phenomenon itself is a known, current
limitation of this simple mutation scheme, not eliminated, and
held-out validation in tests/test_co_evolution.py checks for
(imperfect, realistic) generalization rather than assuming a perfect
result.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from utils.seeding import get_rng
from core.ast_mutator import ASTMutator
from core.morphology import Morphology, MorphologyMutator


@dataclass
class CoEvolvedIndividual:
    source: str
    morphology: Morphology
    fitness: float = -1.0
    task_score: float = 0.0
    hardware_cost: float = 0.0


DEFAULT_SEED_CODE = (
    "def decide(distance_to_target, distance_to_obstacle, sensor_range, max_speed):\n"
    "    if distance_to_obstacle < sensor_range:\n"
    "        return -max_speed\n"
    "    return max_speed\n"
)


def _simulate_episode(source: str, morphology: Morphology, target_dist: float = 20.0,
                       obstacle_dist: float = 10.0, max_steps: int = 60) -> dict:
    """Runs the decide() function in a tiny 1D physics loop. Returns
    whether the target was reached, whether the obstacle was hit, and
    steps taken - the raw ingredients for task_score."""
    namespace = {"__builtins__": {"abs": abs, "max": max, "min": min}}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace)
    except Exception:
        return {"reached_target": False, "hit_obstacle": False, "steps": max_steps, "error": True}

    fn = namespace.get("decide")
    if fn is None:
        return {"reached_target": False, "hit_obstacle": False, "steps": max_steps, "error": True}

    pos, vel = 0.0, 0.0
    for step in range(max_steps):
        dist_to_target = target_dist - pos
        dist_to_obstacle = obstacle_dist - pos
        try:
            accel = fn(dist_to_target, dist_to_obstacle, morphology.sensor_range, morphology.max_speed)
            accel = max(-morphology.max_speed, min(morphology.max_speed, float(accel)))
        except Exception:
            return {"reached_target": False, "hit_obstacle": False, "steps": step, "error": True}

        vel = accel  # simplified: acceleration directly sets velocity (no inertia) for a tractable task
        pos += vel

        if abs(pos - obstacle_dist) < 0.5:
            return {"reached_target": False, "hit_obstacle": True, "steps": step, "error": False}
        if pos >= target_dist:
            return {"reached_target": True, "hit_obstacle": False, "steps": step, "error": False}

    return {"reached_target": False, "hit_obstacle": False, "steps": max_steps, "error": False}


def evaluate_co_fitness(source: str, morphology: Morphology, n_trials: int = 9) -> tuple[float, float, float]:
    """Returns (fitness, task_score, hardware_cost).

    Uses a FIXED grid of obstacle positions (not random sampling) -
    this was changed after finding that random sampling with few trials
    let a non-robust solution ("ignore the sensor, always move at max
    speed") score perfectly just by chance, because the 5 randomly
    sampled obstacle positions happened not to land where that
    constant-speed trajectory would actually collide. A fixed grid
    spanning the task's obstacle range removes that evaluation-noise
    loophole - fitness now reflects performance across the range, not
    luck in which positions got sampled."""
    obstacle_positions = [5.0 + i * (10.0 / (n_trials - 1)) for i in range(n_trials)]
    successes, hits = 0, 0
    for obstacle_dist in obstacle_positions:
        result = _simulate_episode(source, morphology, target_dist=20.0, obstacle_dist=obstacle_dist)
        if result["error"]:
            continue
        if result["reached_target"]:
            successes += 1
        if result["hit_obstacle"]:
            hits += 1

    task_score = (successes - hits) / n_trials  # in [-1, 1]
    hw_cost = morphology.hardware_cost()
    fitness = task_score - 0.05 * hw_cost  # cost penalty - tune-able trade-off weight
    return fitness, task_score, hw_cost


class CoEvolutionEngine:
    def __init__(self, population_size: int = 20, seed_stream: str = "co_evolution"):
        self.population_size = population_size
        self.code_mutator = ASTMutator(seed_stream=f"{seed_stream}_code")
        self.morph_mutator = MorphologyMutator(seed_stream=f"{seed_stream}_morph")
        self.rng = get_rng(seed_stream)

        self.population: list[CoEvolvedIndividual] = [
            CoEvolvedIndividual(source=DEFAULT_SEED_CODE, morphology=self.morph_mutator.random_morphology())
            for _ in range(population_size)
        ]

    def _evaluate_all(self) -> None:
        for ind in self.population:
            fitness, task_score, hw_cost = evaluate_co_fitness(ind.source, ind.morphology)
            ind.fitness, ind.task_score, ind.hardware_cost = fitness, task_score, hw_cost

    def run(self, generations: int = 30, elite_fraction: float = 0.2, verbose: bool = True) -> list[dict]:
        self._evaluate_all()
        history = []

        for gen in range(generations):
            ranked = sorted(self.population, key=lambda i: i.fitness, reverse=True)
            best = ranked[0]
            mean_fitness = sum(i.fitness for i in self.population) / len(self.population)
            mean_sensor = sum(i.morphology.sensor_range for i in self.population) / len(self.population)
            history.append({
                "gen": gen, "best_fitness": best.fitness, "mean_fitness": mean_fitness,
                "best_task_score": best.task_score, "best_hw_cost": best.hardware_cost,
                "mean_sensor_range": mean_sensor,
            })
            if verbose:
                print(f"gen {gen:3d} | best_fitness={best.fitness:+.3f} task_score={best.task_score:+.3f} "
                      f"hw_cost={best.hardware_cost:.3f} sensor_range={best.morphology.sensor_range:.2f} "
                      f"max_speed={best.morphology.max_speed:.2f}")

            n_elite = max(1, int(self.population_size * elite_fraction))
            next_gen = list(ranked[:n_elite])

            while len(next_gen) < self.population_size:
                parent_idx = self.rng.integers(len(ranked[: max(3, n_elite * 2)]))
                parent = ranked[parent_idx]

                # co-mutation: each child mutates EITHER code OR morphology
                # (not both at once), so we can see which axis evolution
                # actually uses to improve fitness.
                if self.rng.random() < 0.5:
                    new_source, _ = self.code_mutator.mutate_source(parent.source)
                    new_morph = parent.morphology
                else:
                    new_source = parent.source
                    new_morph = self.morph_mutator.mutate(parent.morphology)

                next_gen.append(CoEvolvedIndividual(source=new_source, morphology=new_morph))

            self.population = next_gen
            self._evaluate_all()

        return history