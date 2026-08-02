import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.seeding import set_global_seed
from core.morphology import Morphology, MorphologyMutator
from core.co_evolution import CoEvolutionEngine, evaluate_co_fitness, _simulate_episode


def test_morphology_mutation_stays_in_bounds():
    set_global_seed(0)
    mutator = MorphologyMutator()
    m = Morphology()
    for _ in range(500):
        m = mutator.mutate(m)
        lo, hi = mutator.BOUNDS["sensor_range"]
        assert lo <= m.sensor_range <= hi
        lo, hi = mutator.BOUNDS["max_speed"]
        assert lo <= m.max_speed <= hi


def test_hardware_cost_increases_with_capability():
    cheap = Morphology(sensor_range=0.5, max_speed=0.2, comm_range=1.0)
    expensive = Morphology(sensor_range=10.0, max_speed=3.0, comm_range=15.0)
    assert expensive.hardware_cost() > cheap.hardware_cost()


def test_co_evolution_improves_fitness_over_generations():
    set_global_seed(0)
    engine = CoEvolutionEngine(population_size=20)
    history = engine.run(generations=25, verbose=False)
    # Fitness should not get worse - elitism guarantees this structurally,
    # but this catches a regression if elitism logic ever breaks.
    assert history[-1]["best_fitness"] >= history[0]["best_fitness"]


def test_co_evolution_drives_down_sensor_range_when_task_is_easy():
    """The core co-evolution claim: given a cost penalty on hardware,
    evolution should NOT max out sensor_range - it should find the
    minimum that still lets the task be solved reasonably. This is a
    directional check (sensor_range trends down from a random start,
    not that it hits a specific number), since the exact optimum
    depends on the noisy interaction between code bloat and physics."""
    set_global_seed(1)
    engine = CoEvolutionEngine(population_size=20)
    initial_mean_sensor = sum(i.morphology.sensor_range for i in engine.population) / len(engine.population)
    history = engine.run(generations=25, verbose=False)
    final_mean_sensor = history[-1]["mean_sensor_range"]
    assert final_mean_sensor <= initial_mean_sensor, \
        "hardware cost pressure should push sensor_range down or hold it, not increase it unboundedly"


def test_simulate_episode_handles_broken_code_gracefully():
    """A candidate with a syntax error or missing function must not
    crash the evaluation loop - same defensive requirement as
    core/fitness_scorer.py for the original (non-morphological) engine."""
    result = _simulate_episode("this is not valid python(((", Morphology())
    assert result["error"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))