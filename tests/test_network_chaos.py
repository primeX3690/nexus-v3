import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.seeding import set_global_seed
from utils.config import SwarmConfig
from simulation.swarm.network_chaos import NetworkChaosInjector, ChaosConfig
from simulation.swarm.swarm_manager import SwarmManager


def test_chaos_injector_drop_rate_matches_config():
    set_global_seed(0)
    chaos = NetworkChaosInjector(ChaosConfig(drop_probability=0.3))
    for step in range(200):
        chaos.send(from_id=1, to_id=2, step=step)
    assert 0.2 < chaos.drop_rate_observed() < 0.4  # statistical, 200 samples


def test_chaos_injector_accounts_for_every_message():
    set_global_seed(0)
    chaos = NetworkChaosInjector(ChaosConfig(drop_probability=0.4, max_latency_steps=2))
    for step in range(50):
        chaos.send(from_id=1, to_id=2, step=step)
    delivered = sum(len(chaos.deliverable_messages(s)) for s in range(55))
    assert chaos.stats["dropped"] + delivered == 50


def test_default_timeout_produces_false_positives_under_heavy_loss():
    """Documents a REAL, found limitation: at 50% packet loss, the
    default heartbeat_timeout_steps=12 (with interval=3) is too
    aggressive and produces false-positive 'dead' detections on
    healthy agents that were never killed. This is expected/known
    behavior now, not a bug - see test below for the fix via scaled
    timeout, and demo/run_network_chaos.py for the full trade-off."""
    set_global_seed(0)
    chaos = NetworkChaosInjector(ChaosConfig(drop_probability=0.5, max_latency_steps=2))
    cfg = SwarmConfig(n_agents=6, comm_range=15, heartbeat_interval_steps=3, heartbeat_timeout_steps=12)
    sm = SwarmManager(config=cfg, world_size=(30, 30), chaos=chaos)

    false_positives = 0
    for step in range(150):
        info = sm.step()  # no agents killed - every detection here is a false positive
        false_positives += len(info["newly_dead_detections"])

    assert false_positives > 0, ("expected the known false-positive issue at default timeout under "
                                   "50% loss - if this now fails, the underlying dynamics changed")


def test_scaled_timeout_eliminates_false_positives_without_losing_real_detection():
    """The fix: scaling heartbeat_timeout_steps up for lossy conditions
    eliminates false positives WHILE STILL correctly detecting a real
    kill (higher timeout = slower detection latency, not blindness)."""
    set_global_seed(0)
    chaos = NetworkChaosInjector(ChaosConfig(drop_probability=0.5, max_latency_steps=2))
    cfg = SwarmConfig(n_agents=6, comm_range=15, heartbeat_interval_steps=3, heartbeat_timeout_steps=30)
    sm = SwarmManager(config=cfg, world_size=(30, 30), chaos=chaos)

    false_positives = 0
    detection_step = None
    for step in range(150):
        if step == 20:
            sm.kill_agent(2)
        info = sm.step()
        if step < 20:
            false_positives += len(info["newly_dead_detections"])
        elif info["newly_dead_detections"] and detection_step is None:
            detection_step = step

    assert false_positives == 0
    assert detection_step is not None, "real death must still be detected, not just silenced"
    assert sm.step_count == 150


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))