"""
simulation/swarm/network_chaos.py

Injects realistic communication failure into the swarm's heartbeat
channel: random packet drops and delayed delivery, modeling the actual
conditions of disaster zones and space (intermittent, lossy links) -
as opposed to the clean, always-delivered heartbeats
fault_tolerance.py's original tests assumed.

Design: this wraps the DELIVERY of a heartbeat, not the detection logic
in fault_tolerance.py itself - FaultToleranceManager doesn't need to
know chaos exists, it just sometimes receives fewer/later heartbeats
than were actually sent, exactly like a real lossy network. This keeps
two concerns cleanly separated: fault_tolerance.py answers "is this
peer still alive," network_chaos.py answers "did this message arrive."
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.seeding import get_rng


@dataclass
class ChaosConfig:
    drop_probability: float = 0.2      # 0.0-1.0, fraction of packets silently lost
    min_latency_steps: int = 0          # extra delay before delivery, in simulation steps
    max_latency_steps: int = 3


class NetworkChaosInjector:
    """Sits between 'agent sends heartbeat' and 'peer receives it'.
    Call `send(from_id, to_id, step)` when a message is sent; call
    `deliverable_messages(current_step)` each tick to get back the
    messages that should actually arrive THIS step (accounting for
    drops and delay) - some sent messages never appear here at all."""

    def __init__(self, config: ChaosConfig | None = None, seed_stream: str = "network_chaos"):
        self.config = config or ChaosConfig()
        self.rng = get_rng(seed_stream)
        self._in_flight: list[dict] = []
        self.stats = {"sent": 0, "dropped": 0, "delivered": 0}

    def send(self, from_id: int, to_id: int, step: int) -> None:
        self.stats["sent"] += 1
        if self.rng.random() < self.config.drop_probability:
            self.stats["dropped"] += 1
            return  # silently lost - never scheduled for delivery, mirrors a real dropped packet

        delay = int(self.rng.integers(self.config.min_latency_steps, self.config.max_latency_steps + 1))
        self._in_flight.append({
            "from_id": from_id, "to_id": to_id,
            "sent_step": step, "deliver_step": step + delay,
        })

    def deliverable_messages(self, current_step: int) -> list[dict]:
        """Call once per simulation step. Returns messages that arrive
        THIS step and removes them from the in-flight queue."""
        ready = [m for m in self._in_flight if m["deliver_step"] <= current_step]
        self._in_flight = [m for m in self._in_flight if m["deliver_step"] > current_step]
        self.stats["delivered"] += len(ready)
        return ready

    def drop_rate_observed(self) -> float:
        return self.stats["dropped"] / self.stats["sent"] if self.stats["sent"] > 0 else 0.0