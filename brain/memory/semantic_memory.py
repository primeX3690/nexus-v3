"""
brain/memory/semantic_memory.py

Facts and learned generalizations extracted from experience - e.g.
"action 2 near obstacles tends to fail" - stored as simple
(key -> value, confidence) triples that strengthen or weaken with
repeated evidence. This is what the symbolic reasoning engine's
condition/action rules can be dynamically generated from over time,
as opposed to the hardcoded safety rules in safety/constitutional_ai.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Fact:
    key: str
    value: object
    confidence: float = 0.5   # 0..1, updated via simple exponential evidence accumulation
    observations: int = 0


class SemanticMemory:
    def __init__(self, learning_rate: float = 0.2):
        self.facts: dict[str, Fact] = {}
        self.learning_rate = learning_rate

    def observe(self, key: str, value: object, supports: bool = True) -> None:
        """Record one piece of evidence. `supports=False` means this
        observation contradicts the current believed value, and pulls
        confidence down rather than up."""
        if key not in self.facts:
            self.facts[key] = Fact(key=key, value=value, confidence=0.5, observations=1)
            return

        fact = self.facts[key]
        fact.observations += 1
        if value == fact.value and supports:
            fact.confidence += self.learning_rate * (1 - fact.confidence)
        else:
            fact.confidence -= self.learning_rate * fact.confidence
            if fact.confidence < 0.3:
                # Belief has been undermined enough - replace it with the new value.
                fact.value = value
                fact.confidence = 0.5

    def query(self, key: str) -> Fact | None:
        """Returns a SNAPSHOT (copy) of the fact, not the live internal
        object - observe() mutates facts in place, so returning the
        live reference would let a caller's earlier `query()` result
        silently change value out from under it, and would let external
        code corrupt internal state by mutating the returned object."""
        fact = self.facts.get(key)
        if fact is None:
            return None
        return replace(fact)

    def confident_facts(self, threshold: float = 0.7) -> dict[str, object]:
        return {k: f.value for k, f in self.facts.items() if f.confidence >= threshold}

    def as_rule_facts(self) -> dict:
        """Export high-confidence beliefs in a flat dict, ready to feed
        into SymbolicEngine.update_facts()."""
        return self.confident_facts()