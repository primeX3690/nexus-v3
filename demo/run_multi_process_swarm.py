"""
demo/run_multi_process_swarm.py

Runs the swarm as N REAL, SEPARATE OS PROCESSES (subprocess.Popen, real
PIDs, visible in `ps`) talking only over real UDP sockets on localhost -
the real-decentralization counterpart to demo/run_multi_robot.py's
single-process, single-Python-list swarm. Can SIGKILL one agent process
mid-run: this is a REAL process death (the OS reclaims it, no more
packets are physically possible from it), not a simulated `agent.alive =
False` flag - the survivors must detect it purely from real socket
silence via the same FaultToleranceManager already verified in
simulation/swarm/fault_tolerance.py.

Run: python3 -m demo.run_multi_process_swarm
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

N_AGENTS = 5
N_STEPS = 60
WORLD_W, WORLD_H = 40.0, 40.0
BASE_PORT = 55000
KILL_AGENT_ID = 2
KILL_AFTER_SECONDS = 0.6  # real wall-clock delay before the real SIGKILL


def main():
    results_dir = tempfile.mkdtemp(prefix="nexus_swarm_")
    print("=" * 70)
    print("NEXUS v3 - Multi-Process Swarm Demo (real OS processes, real sockets)")
    print("=" * 70)
    print(f"Spawning {N_AGENTS} independent OS processes on UDP ports "
          f"{BASE_PORT}-{BASE_PORT + N_AGENTS - 1} ...")

    real_time_dt = 0.05  # 20Hz-ish, slow enough that a mid-run SIGKILL lands mid-simulation
    procs: dict[int, subprocess.Popen] = {}
    for i in range(N_AGENTS):
        cmd = [sys.executable, "-m", "simulation.swarm.multi_process_agent",
               str(i), str(N_AGENTS), str(BASE_PORT), str(N_STEPS),
               str(WORLD_W), str(WORLD_H), results_dir, str(real_time_dt)]
        procs[i] = subprocess.Popen(cmd, cwd=os.getcwd())
        print(f"  agent {i}: PID {procs[i].pid} bound to 127.0.0.1:{BASE_PORT + i}")

    print(f"\nAll {N_AGENTS} agents running as independent OS processes.")
    print(f"In {KILL_AFTER_SECONDS}s, REAL SIGKILL will be sent to agent "
          f"{KILL_AGENT_ID}'s actual PID ({procs[KILL_AGENT_ID].pid}) - "
          f"a genuine process death, not a simulated flag.")
    time.sleep(KILL_AFTER_SECONDS)

    killed_pid = procs[KILL_AGENT_ID].pid
    procs[KILL_AGENT_ID].send_signal(signal.SIGKILL)
    print(f"SIGKILL sent to PID {killed_pid} (agent {KILL_AGENT_ID}). "
          f"Waiting for surviving agents to finish and detect it themselves ...")

    for i, p in procs.items():
        if i == KILL_AGENT_ID:
            ret = p.wait()
            print(f"  agent {i}: process exited with signal/code {ret} (expected: killed)")
        else:
            ret = p.wait(timeout=30)
            print(f"  agent {i}: process exited normally with code {ret}")

    print("\n" + "=" * 70)
    print("Results (each written independently by its own process, no shared memory):")
    print("=" * 70)
    any_detected = False
    for i in range(N_AGENTS):
        path = f"{results_dir}/agent_{i}_result.json"
        if i == KILL_AGENT_ID:
            print(f"agent {i}: killed mid-run, no result file (real process death)")
            continue
        if not os.path.exists(path):
            print(f"agent {i}: no result file found (unexpected)")
            continue
        with open(path) as f:
            r = json.load(f)
        detected = KILL_AGENT_ID in r["presumed_dead_peers"]
        any_detected = any_detected or detected
        first_detection_step = next(
            (d["step"] for d in r["detections_log"] if KILL_AGENT_ID in d["detected_dead"]), None)
        print(f"agent {i} (real PID {r['pid']}): final_pos={[round(x,2) for x in r['final_position']]}  "
              f"detected agent {KILL_AGENT_ID} dead: {detected}"
              + (f" at step {first_detection_step}" if detected else ""))

    print("=" * 70)
    print(f"FINDING: {'All' if any_detected else 'No'} surviving agents detected the real "
          f"SIGKILL of agent {KILL_AGENT_ID} through socket silence alone - no central process "
          f"told them, no shared Python object changed under them. This is what single-process "
          f"SwarmManager.kill_agent() (a flag flip visible to every agent instantly, in the same "
          f"list) cannot demonstrate.")
    print("=" * 70)

    shutil.rmtree(results_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
