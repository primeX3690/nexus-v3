"""
Centralized seeding so every module (genetic engine, SNN, PPO, swarm)
draws from reproducible, independently-streamed RNGs.

Why this matters for NEXUS specifically: the genetic engine, the PPO
agent, and the swarm manager all sample randomness in the same process.
If they all pull from the *same* global numpy RNG, changing the number
of swarm agents shifts the genetic engine's mutation sequence and makes
runs non-reproducible in a way that's very hard to debug. Using
`np.random.default_rng` with `spawn()` gives each subsystem its own
independent stream from one master seed.
"""
from __future__ import annotations

import hashlib
import os
import random
import numpy as np

_MASTER_SEED: int | None = None
_MASTER_RNG: np.random.Generator | None = None


def set_global_seed(seed: int = 42) -> None:
    """Seed python's `random`, numpy legacy global state, and the master
    Generator used to spawn per-subsystem RNGs. Call this once at process
    start (e.g. top of demo/run_*.py)."""
    global _MASTER_SEED, _MASTER_RNG
    _MASTER_SEED = seed
    random.seed(seed)
    np.random.seed(seed)  # legacy global state, some libs still read this
    _MASTER_RNG = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_rng(stream_name: str = "default") -> np.random.Generator:
    """Return an independent RNG stream derived from the master seed.

    Each subsystem should request its own named stream, e.g.:
        rng = get_rng("genetic_engine")
        rng = get_rng("ppo_agent")
    This avoids cross-contamination between subsystems while staying
    fully reproducible from a single `set_global_seed` call.
    """
    global _MASTER_RNG
    if _MASTER_RNG is None:
        set_global_seed(42)
    # Deterministic per-stream seed derivation. NOTE: Python's built-in
    # hash() is randomized per-process for strings (PYTHONHASHSEED),
    # and setting the env var mid-process does NOT retroactively fix
    # the current interpreter - so we use hashlib (stable across
    # processes and platforms) instead of hash().
    digest = hashlib.sha256(f"{_MASTER_SEED}:{stream_name}".encode()).digest()
    stream_seed = int.from_bytes(digest[:4], byteorder="big")
    return np.random.default_rng(stream_seed)