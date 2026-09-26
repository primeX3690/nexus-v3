"""
tests/test_multi_process_swarm.py

Verifies the multi-process swarm agent (simulation/swarm/multi_process_agent.py)
runs correctly as a real OS subprocess and produces a valid result file -
a lighter-weight version of demo/run_multi_process_swarm.py's full
SIGKILL scenario (kept fast for CI: 3 agents, 15 steps, no real-time
sleep, no mid-run kill - just verifying the real-subprocess + real-UDP
plumbing itself works end to end).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest


def test_multi_process_agents_run_and_produce_results():
    """real_time_dt is deliberately non-zero here (0.02s/step) rather than
    0.0 ('run flat out'). A REAL finding from writing this test: at
    real_time_dt=0.0, independent OS processes are scheduled by the OS at
    slightly different real rates with nothing forcing them into lockstep
    (there is no global clock in a genuinely decentralized design), so a
    faster process can race past its own heartbeat-timeout step count
    before a slower peer has sent more than one or two heartbeats -
    producing a genuine false "peer presumed dead" detection with nobody
    actually killed. This is the SAME class of false-positive already
    documented for packet loss in demo/run_network_chaos.py, just from
    process-scheduling jitter instead of a lossy link - both are real
    consequences of true decentralization, not something to hide by
    quietly picking a timing that happens not to trigger it."""
    n_agents = 3
    n_steps = 15
    base_port = 56500
    with tempfile.TemporaryDirectory() as results_dir:
        procs = []
        for i in range(n_agents):
            cmd = [sys.executable, "-m", "simulation.swarm.multi_process_agent",
                   str(i), str(n_agents), str(base_port), str(n_steps),
                   "30.0", "30.0", results_dir, "0.02"]
            procs.append(subprocess.Popen(cmd, cwd=os.getcwd()))

        for p in procs:
            ret = p.wait(timeout=30)
            assert ret == 0, "each agent process must exit cleanly with no kill involved"

        for i in range(n_agents):
            path = os.path.join(results_dir, f"agent_{i}_result.json")
            assert os.path.exists(path), f"agent {i} did not write a result file"
            with open(path) as f:
                result = json.load(f)
            assert result["agent_id"] == i
            assert len(result["final_position"]) == 2
            assert result["pid"] != os.getpid(), "agent must run as its own OS process, not inline"
            # with no kill and only 3 agents at close range on a small map,
            # nobody should have been falsely presumed dead
            assert result["presumed_dead_peers"] == []


def test_multi_process_swarm_detects_real_sigkill():
    """Faster variant of demo/run_multi_process_swarm.py's scenario:
    really SIGKILLs one real subprocess mid-run and checks survivors
    detect it via real socket silence, not a shared-memory flag."""
    import signal
    import time

    n_agents = 4
    n_steps = 40
    base_port = 56600
    kill_id = 1
    with tempfile.TemporaryDirectory() as results_dir:
        procs = {}
        for i in range(n_agents):
            cmd = [sys.executable, "-m", "simulation.swarm.multi_process_agent",
                   str(i), str(n_agents), str(base_port), str(n_steps),
                   "30.0", "30.0", results_dir, "0.03"]
            procs[i] = subprocess.Popen(cmd, cwd=os.getcwd())

        time.sleep(0.4)
        killed_pid = procs[kill_id].pid
        procs[kill_id].send_signal(signal.SIGKILL)

        for i, p in procs.items():
            ret = p.wait(timeout=30)
            if i == kill_id:
                assert ret == -signal.SIGKILL, "the killed agent's real exit status must reflect SIGKILL"

        for i in range(n_agents):
            if i == kill_id:
                continue
            path = os.path.join(results_dir, f"agent_{i}_result.json")
            with open(path) as f:
                result = json.load(f)
            assert kill_id in result["presumed_dead_peers"], (
                f"surviving agent {i} (real PID {result['pid']}) failed to detect the real "
                f"SIGKILL of PID {killed_pid} through socket silence"
            )


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
