"""
simulation/swarm/fault_tolerance.py

Decentralized fault detection: every agent periodically broadcasts a
heartbeat; every OTHER agent independently tracks when it last heard
from each peer. If a peer goes silent for longer than a timeout, agents
independently mark it dead and the swarm re-forms without it - no
central coordinator, so there's no single point of failure for fault
detection itself (a naive "manager pings everyone" design would just
move the SPOF problem, not remove it).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HeartbeatState:
    last_seen_step: dict[int, int] = field(default_factory=dict)
    presumed_dead: set[int] = field(default_factory=set)


class FaultToleranceManager:
    def __init__(self, agent_id: int, heartbeat_interval_steps: int = 5,
                 heartbeat_timeout_steps: int = 15):
        self.agent_id = agent_id
        self.heartbeat_interval = heartbeat_interval_steps
        self.heartbeat_timeout = heartbeat_timeout_steps
        self.state = HeartbeatState()
        self._current_step = 0

    def should_broadcast(self, step: int) -> bool:
        return step % self.heartbeat_interval == 0

    def receive_heartbeat(self, from_agent_id: int, step: int) -> None:
        if from_agent_id == self.agent_id:
            return
        self.state.last_seen_step[from_agent_id] = step
        self.state.presumed_dead.discard(from_agent_id)  # a late heartbeat resurrects a peer

    def tick(self, step: int, known_peer_ids: set[int]) -> set[int]:
        """Call once per simulation step with the full set of peer ids
        that are SUPPOSED to exist. Returns the set of newly-detected
        failures this tick (peers that just crossed the timeout)."""
        self._current_step = step
        newly_dead = set()
        for peer_id in known_peer_ids:
            if peer_id == self.agent_id:
                continue
            last_seen = self.state.last_seen_step.get(peer_id, None)
            if last_seen is None:
                # Never heard from them at all - only declare dead once
                # they've had a fair chance (one full timeout window from
                # step 0), not immediately at step 0.
                if step >= self.heartbeat_timeout and peer_id not in self.state.presumed_dead:
                    self.state.presumed_dead.add(peer_id)
                    newly_dead.add(peer_id)
                continue
            silence = step - last_seen
            if silence > self.heartbeat_timeout and peer_id not in self.state.presumed_dead:
                self.state.presumed_dead.add(peer_id)
                newly_dead.add(peer_id)
        return newly_dead

    def alive_peers(self, known_peer_ids: set[int]) -> set[int]:
        return {p for p in known_peer_ids if p != self.agent_id and p not in self.state.presumed_dead}