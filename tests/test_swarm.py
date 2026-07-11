import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from utils.seeding import set_global_seed
from utils.config import SwarmConfig
from simulation.swarm.stigmergy_map import StigmergyMap
from simulation.swarm.fault_tolerance import FaultToleranceManager
from simulation.swarm.swarm_manager import SwarmManager


def test_stigmergy_deposit_and_evaporate():
    sm = StigmergyMap(10, 10, decay_rate=0.1)
    sm.deposit(5, 5, amount=10.0)
    sm.evaporate()
    assert np.isclose(sm.read(5, 5), 9.0)


def test_fault_tolerance_detects_silent_peer():
    peers = {1, 2, 3}
    ft = FaultToleranceManager(agent_id=0, heartbeat_interval_steps=5, heartbeat_timeout_steps=10)
    for step in range(30):
        for p in [1, 2]:
            if step % 5 == 0:
                ft.receive_heartbeat(p, step)
        ft.tick(step, peers)
    assert 3 not in ft.alive_peers(peers)
    assert {1, 2} <= ft.alive_peers(peers)


def test_swarm_stays_within_world_bounds():
    set_global_seed(0)
    sm = SwarmManager(config=SwarmConfig(n_agents=6, comm_range=15), world_size=(30, 30))
    for _ in range(60):
        info = sm.step()
    for a in sm.agents:
        assert 0 <= a.position[0] <= 30
        assert 0 <= a.position[1] <= 30


def test_swarm_fault_detection_reduces_alive_count():
    set_global_seed(0)
    sm = SwarmManager(config=SwarmConfig(n_agents=5, heartbeat_timeout_steps=10), world_size=(20, 20))
    for t in range(30):
        info = sm.step()
        if t == 5:
            sm.kill_agent(1)
    assert info["alive_count"] == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))