"""
Typed configuration for NEXUS v3.

Deliberately uses only stdlib (dataclasses + json) - no PyYAML - so the
whole project stays installable on a fresh machine with just
numpy/scipy/pytest. If you later want YAML configs, `load_config` is the
only function that would need to change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SNNConfig:
    n_neurons: int = 128
    tau_mem: float = 20.0       # membrane time constant (ms)
    tau_syn: float = 5.0        # synaptic time constant (ms)
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 1.0             # simulation timestep (ms)
    refractory_steps: int = 2


@dataclass
class MambaConfig:
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    epochs_per_update: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class GeneticConfig:
    population_size: int = 30
    mutation_rate: float = 0.15
    elite_fraction: float = 0.2
    tournament_size: int = 3
    max_generations: int = 50


@dataclass
class SwarmConfig:
    n_agents: int = 5
    comm_range: float = 10.0
    heartbeat_interval_steps: int = 5
    heartbeat_timeout_steps: int = 15


@dataclass
class NexusConfig:
    seed: int = 42
    device: str = "cpu"           # always cpu - no GPU path in this project
    snn: SNNConfig = field(default_factory=SNNConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def load_config(path: str | Path | None = None) -> NexusConfig:
    """Load a NexusConfig from a JSON file, or return defaults if no
    path is given / file doesn't exist yet."""
    if path is None:
        return NexusConfig()
    p = Path(path)
    if not p.exists():
        return NexusConfig()
    raw = json.loads(p.read_text())
    return NexusConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "cpu"),
        snn=SNNConfig(**raw.get("snn", {})),
        mamba=MambaConfig(**raw.get("mamba", {})),
        ppo=PPOConfig(**raw.get("ppo", {})),
        genetic=GeneticConfig(**raw.get("genetic", {})),
        swarm=SwarmConfig(**raw.get("swarm", {})),
    )