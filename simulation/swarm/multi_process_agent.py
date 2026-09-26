"""
simulation/swarm/multi_process_agent.py

A swarm agent that runs as its OWN OS PROCESS (spawned by
run_multi_process_swarm.py via subprocess.Popen, each with its own PID),
and talks to every other agent SOLELY over real UDP sockets on localhost
- no shared Python objects, no shared memory, no in-process function
calls between agents. This is the real decentralization proof the
single-process SwarmManager (which just holds every agent's object in
one Python list) cannot offer: killing this process for real (SIGKILL,
not a `agent.alive = False` flag flip) removes it from the OS process
table entirely, and survivors must detect the failure through real
socket silence, exactly like a real multi-robot deployment across
separate physical machines (or separate onboard computers) would.

Reuses the EXACT SAME verified logic as the single-process version:
SwarmAgent (boids + stigmergy pull), StigmergyMap, FaultToleranceManager
- only the transport between agents changes, not the swarm intelligence
itself, so this is a real decentralization test of the same algorithm,
not a rewrite.

Each agent maintains its own LOCAL stigmergy map (a real distributed-
systems design choice, not a shortcut): there is no shared map object to
reach across processes, so each agent deposits into its own map and
broadcasts its deposit to peers, who fold it into their own local copies
- eventually-consistent stigmergy, same principle real ant colonies /
decentralized robot swarms use (no robot has a global map either).
"""
from __future__ import annotations

import json
import socket
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, ".")
from simulation.swarm.swarm_agent import SwarmAgent  # noqa: E402
from simulation.swarm.stigmergy_map import StigmergyMap  # noqa: E402
from simulation.swarm.fault_tolerance import FaultToleranceManager  # noqa: E402

HEARTBEAT_INTERVAL_STEPS = 3
HEARTBEAT_TIMEOUT_STEPS = 10
RECV_BUFSIZE = 4096


@dataclass
class PeerView:
    """Duck-typed stand-in for a remote SwarmAgent, built purely from the
    last UDP packet received from that peer - SwarmAgent._neighbors() and
    _boids_forces() only ever touch .agent_id/.alive/.position/.velocity,
    so this is a legitimate substitute, not a hack."""
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    alive: bool = True


def main():
    agent_id = int(sys.argv[1])
    n_agents = int(sys.argv[2])
    base_port = int(sys.argv[3])
    n_steps = int(sys.argv[4])
    world_w = float(sys.argv[5])
    world_h = float(sys.argv[6])
    results_dir = sys.argv[7]
    real_time_dt = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

    rng = np.random.default_rng(agent_id)  # each process seeds independently, by its own id
    start_pos = rng.uniform(0, min(world_w, world_h), size=2)

    agent = SwarmAgent(agent_id=agent_id, position=start_pos, perception_radius=15.0)
    stigmergy = StigmergyMap(width=int(world_w), height=int(world_h))
    fault_mgr = FaultToleranceManager(agent_id=agent_id,
                                       heartbeat_interval_steps=HEARTBEAT_INTERVAL_STEPS,
                                       heartbeat_timeout_steps=HEARTBEAT_TIMEOUT_STEPS)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", base_port + agent_id))
    sock.setblocking(False)

    peer_ports = {i: base_port + i for i in range(n_agents) if i != agent_id}
    peer_last_state: dict[int, dict] = {}
    all_ids = set(range(n_agents))

    detections_log = []

    for step in range(n_steps):
        # --- broadcast own state as a real UDP datagram to every peer ---
        payload = json.dumps({
            "agent_id": agent_id,
            "step": step,
            "position": agent.position.tolist(),
            "velocity": agent.velocity.tolist(),
            "alive": agent.alive,
            "heartbeat": fault_mgr.should_broadcast(step),
        }).encode()
        for pid, port in peer_ports.items():
            try:
                sock.sendto(payload, ("127.0.0.1", port))
            except OSError:
                pass  # a peer that already exited is a real, expected condition, not an error

        # --- drain the real inbound socket queue (non-blocking) ---
        while True:
            try:
                data, _addr = sock.recvfrom(RECV_BUFSIZE)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            from_id = msg["agent_id"]
            peer_last_state[from_id] = msg
            if msg.get("heartbeat"):
                fault_mgr.receive_heartbeat(from_id, msg["step"])
            # fold the peer's stigmergy deposit into our own local map -
            # eventually-consistent, no shared object.
            gx = int(np.clip(msg["position"][0], 0, world_w - 1))
            gy = int(np.clip(msg["position"][1], 0, world_h - 1))
            stigmergy.deposit(gx, gy, amount=0.5)

        newly_dead = fault_mgr.tick(step, all_ids)
        if newly_dead:
            detections_log.append({"step": step, "detected_dead": sorted(newly_dead)})

        # build the local swarm view purely from real received packets
        peers = [
            PeerView(agent_id=pid, position=np.array(s["position"]),
                     velocity=np.array(s["velocity"]), alive=s.get("alive", True))
            for pid, s in peer_last_state.items()
            if pid not in fault_mgr.state.presumed_dead
        ]

        agent.step([agent] + peers, stigmergy=stigmergy, world_size=(world_w, world_h))
        gx = int(np.clip(agent.position[0], 0, world_w - 1))
        gy = int(np.clip(agent.position[1], 0, world_h - 1))
        stigmergy.deposit(gx, gy, amount=0.5)
        stigmergy.evaporate()

        if real_time_dt > 0:
            time.sleep(real_time_dt)

    result = {
        "agent_id": agent_id,
        "final_position": agent.position.tolist(),
        "alive": agent.alive,
        "presumed_dead_peers": sorted(fault_mgr.state.presumed_dead),
        "detections_log": detections_log,
        "pid": __import__("os").getpid(),
    }
    with open(f"{results_dir}/agent_{agent_id}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    sock.close()


if __name__ == "__main__":
    main()
