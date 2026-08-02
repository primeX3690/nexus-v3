"""
core/morphology.py

Extends the genetic engine from evolving CODE only to evolving
CODE + PHYSICAL PARAMETERS together ("morphological co-evolution").

Thesis being tested: a fixed-cost budget can be spent on better sensors/
actuators (hardware) OR on smarter control code (software) - evolution
should find where the trade-off actually lies for a given task, rather
than us hand-picking hardware specs. This directly matters for the
resource-constrained robotics story: is it better to evolve a bigger
sensor_range with dumb code, or a small sensor_range with clever code?

Genome = (source_code: str, morphology: Morphology). AST mutation
(core/ast_mutator.py) evolves the code half; MorphologyMutator (below)
evolves the physical half. Both are scored by ONE fitness function that
includes a hardware "cost" term, so evolution is pressured to find an
efficient point on the cost/capability curve, not just maximize sensor
range unboundedly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from utils.seeding import get_rng


@dataclass(frozen=True)
class Morphology:
    """Physical parameters of a simulated robot. Bounds are enforced by
    MorphologyMutator, not here, so this stays a plain data container."""
    sensor_range: float = 3.0     # how far it can detect an obstacle
    max_speed: float = 1.0        # how fast it can move per step
    comm_range: float = 5.0       # how far it can coordinate with swarm-mates

    def hardware_cost(self) -> float:
        """Bigger/better hardware costs more - this is what stops
        evolution from just maxing out every parameter. Roughly
        modeled as super-linear in sensor_range (sensors are the
        expensive part on real cheap robots) and linear in the rest."""
        return (self.sensor_range ** 1.5) * 0.15 + self.max_speed * 0.3 + self.comm_range * 0.05


class MorphologyMutator:
    """Gaussian-perturbation mutator with hard bounds, mirroring the
    role core/ast_mutator.py plays for code but for continuous physical
    parameters instead of discrete AST edits."""

    BOUNDS = {
        "sensor_range": (0.5, 10.0),
        "max_speed": (0.2, 3.0),
        "comm_range": (1.0, 15.0),
    }

    def __init__(self, seed_stream: str = "morphology_mutator", mutation_std_frac: float = 0.2):
        self.rng = get_rng(seed_stream)
        self.mutation_std_frac = mutation_std_frac

    def mutate(self, morphology: Morphology) -> Morphology:
        field_names = list(Morphology.__dataclass_fields__.keys())
        target = field_names[int(self.rng.integers(len(field_names)))]
        current_val = getattr(morphology, target)
        low, high = self.BOUNDS[target]

        std = (high - low) * self.mutation_std_frac
        raw = current_val + self.rng.normal(0, std)
        new_val = max(low, min(high, raw))

        return replace(morphology, **{target: new_val})

    def random_morphology(self) -> Morphology:
        return Morphology(
            sensor_range=float(self.rng.uniform(*self.BOUNDS["sensor_range"])),
            max_speed=float(self.rng.uniform(*self.BOUNDS["max_speed"])),
            comm_range=float(self.rng.uniform(*self.BOUNDS["comm_range"])),
        )