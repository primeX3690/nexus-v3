import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from utils.seeding import set_global_seed
from utils.config import SNNConfig, MambaConfig
from brain.perception.snn_layer import LIFLayer
from brain.world_model.mamba_ssm import MambaSSM
from brain.world_model.world_model import WorldModel
from brain.reasoning.symbolic_engine import SymbolicEngine, Rule
from brain.reasoning.planner import MCTSPlanner


def test_snn_zero_input_never_spikes():
    set_global_seed(0)
    layer = LIFLayer(n_inputs=5, config=SNNConfig(n_neurons=10))
    spikes = layer.run_sequence(np.zeros((30, 5)))
    assert spikes.sum() == 0


def test_snn_strong_input_produces_spikes():
    set_global_seed(0)
    layer = LIFLayer(n_inputs=5, config=SNNConfig(n_neurons=10))
    spikes = layer.run_sequence(np.ones((100, 5)))
    assert spikes.sum() > 0
    assert set(np.unique(spikes)).issubset({0.0, 1.0})


def test_mamba_ssm_stable_no_nan():
    set_global_seed(0)
    ssm = MambaSSM(MambaConfig(d_model=8, d_state=4))
    x = np.random.default_rng(1).normal(0, 1, (500, 8))
    y = ssm.run_sequence(x)
    assert not np.any(np.isnan(y))
    assert np.max(np.abs(y)) < 1e6


def test_world_model_imagine_shape_and_start_state():
    set_global_seed(0)
    wm = WorldModel(state_dim=3, action_dim=2, config=MambaConfig(d_model=8, d_state=4))
    s0 = np.array([1.0, 2.0, 3.0])
    traj = wm.imagine(s0, [np.zeros(2) for _ in range(5)])
    assert traj.shape == (6, 3)
    assert np.allclose(traj[0], s0)


def test_symbolic_engine_priority_conflict_resolution():
    engine = SymbolicEngine()
    engine.set_facts({"battery_pct": 2, "mode": "explore"})
    engine.add_rule(Rule("low", lambda f: f["battery_pct"] < 20 and f.get("mode") not in ("returning", "halted"),
                          lambda f: {"mode": "returning"}, priority=10))
    engine.add_rule(Rule("critical", lambda f: f["battery_pct"] < 5 and f.get("mode") != "halted",
                          lambda f: {"mode": "halted"}, priority=20))
    engine.forward_chain()
    assert engine.facts["mode"] == "halted"


def test_symbolic_engine_detects_oscillation():
    engine = SymbolicEngine()
    engine.set_facts({"x": 0})
    engine.add_rule(Rule("to_one", lambda f: f["x"] == 0, lambda f: {"x": 1}, priority=1))
    engine.add_rule(Rule("to_zero", lambda f: f["x"] == 1, lambda f: {"x": 0}, priority=1))
    fired = engine.forward_chain(max_iterations=20)
    assert any("oscillation" in msg for msg in fired)
    assert len(fired) < 20


def test_mcts_finds_optimal_action():
    set_global_seed(0)

    class ToyWM:
        def reset(self): pass
        def predict_next(self, s, a): return s + a

    actions = [np.array([-1.0]), np.array([0.0]), np.array([1.0])]
    planner = MCTSPlanner(ToyWM(), actions, reward_fn=lambda s, a, ns: float(ns[0]), rollout_depth=3)
    best_idx, root = planner.plan(np.array([0.0]), n_simulations=150)
    assert best_idx == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))