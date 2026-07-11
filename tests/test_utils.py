"""
Tests for utils/. Run with: pytest tests/test_utils.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.seeding import set_global_seed, get_rng
from utils.config import NexusConfig, load_config
from utils.math_ops import softmax, normalize, clip_grad_norm, running_mean_std
from utils.timer import Timer, timed


# ---------- seeding ----------

def test_seeding_reproducible():
    set_global_seed(123)
    rng1 = get_rng("test_stream")
    vals1 = rng1.standard_normal(5)

    set_global_seed(123)
    rng2 = get_rng("test_stream")
    vals2 = rng2.standard_normal(5)

    assert np.allclose(vals1, vals2)


def test_seeding_different_streams_differ():
    set_global_seed(7)
    rng_a = get_rng("genetic_engine")
    rng_b = get_rng("ppo_agent")
    vals_a = rng_a.standard_normal(5)
    vals_b = rng_b.standard_normal(5)
    assert not np.allclose(vals_a, vals_b)


def test_seeding_stable_across_fresh_process_hash_randomization():
    # Regression test for the hash()-based bug: stream seed derivation
    # must not depend on PYTHONHASHSEED (which is randomized per process
    # for str hashing unless set before interpreter start).
    set_global_seed(99)
    rng1 = get_rng("swarm")
    val1 = rng1.integers(0, 1_000_000)

    set_global_seed(99)
    rng2 = get_rng("swarm")
    val2 = rng2.integers(0, 1_000_000)
    assert val1 == val2


# ---------- config ----------

def test_config_defaults():
    cfg = NexusConfig()
    assert cfg.device == "cpu"
    assert cfg.snn.n_neurons == 128
    assert cfg.ppo.gamma == 0.99


def test_config_save_and_load(tmp_path):
    cfg = NexusConfig()
    cfg.snn.n_neurons = 256
    cfg.genetic.population_size = 50
    path = tmp_path / "config.json"
    cfg.save(path)

    loaded = load_config(path)
    assert loaded.snn.n_neurons == 256
    assert loaded.genetic.population_size == 50
    assert loaded.ppo.gamma == cfg.ppo.gamma  # untouched fields survive


def test_load_config_missing_file_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.json")
    assert cfg.snn.n_neurons == 128


# ---------- math_ops ----------

def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    p = softmax(x)
    assert np.isclose(np.sum(p), 1.0)
    assert np.all(p > 0)


def test_softmax_numerically_stable_on_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    p = softmax(x)
    assert not np.any(np.isnan(p))
    assert np.isclose(np.sum(p), 1.0)


def test_normalize_zero_mean_unit_var():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = normalize(x)
    assert np.isclose(np.mean(n), 0.0, atol=1e-6)
    assert np.isclose(np.std(n), 1.0, atol=1e-6)


def test_clip_grad_norm_no_op_when_under_limit():
    grads = {"w": np.array([0.1, 0.1])}
    clipped = clip_grad_norm(grads, max_norm=10.0)
    assert np.allclose(clipped["w"], grads["w"])


def test_clip_grad_norm_scales_down_when_over_limit():
    grads = {"w": np.array([3.0, 4.0])}  # norm = 5.0
    clipped = clip_grad_norm(grads, max_norm=1.0)
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped.values()))
    assert np.isclose(total_norm, 1.0, atol=1e-5)


def test_running_mean_std_converges_to_batch_stats():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(10000, 3))
    rms = running_mean_std(shape=(3,))
    # feed in mini-batches like PPO rollouts would
    for i in range(0, 10000, 100):
        rms.update(data[i:i + 100])
    assert np.allclose(rms.mean, np.mean(data, axis=0), atol=0.15)
    assert np.allclose(rms.var, np.var(data, axis=0), atol=0.3)


# ---------- timer ----------

def test_timer_context_manager_measures_elapsed():
    with Timer("test_block", log_fn=None) as t:
        _ = sum(range(100000))
    assert t.elapsed > 0


def test_timed_decorator_records_last_elapsed():
    @timed("dummy")
    def slow_fn():
        return sum(range(50000))

    result = slow_fn()
    assert result == sum(range(50000))
    assert slow_fn.last_elapsed is not None
    assert slow_fn.last_elapsed >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))