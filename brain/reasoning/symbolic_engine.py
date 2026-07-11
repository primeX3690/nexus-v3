"""
brain/reasoning/symbolic_engine.py

A small forward-chaining production-rule engine: a set of facts (a
knowledge base) plus IF-THEN rules that fire when their conditions are
satisfied, deriving new facts. This is the "symbolic" half of the
neuro-symbolic split - it handles things that are easy to state as
hard logical constraints (e.g. "IF battery_low AND far_from_base THEN
return_to_base") which a neural policy alone tends to violate
occasionally, exactly the failures that matter most in robotics.

No LLM, no learned weights here by design - these rules are meant to be
auditable and provably enforced, which is also why safety/constitutional_ai.py
reuses this same engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Rule:
    name: str
    condition: Callable[[dict], bool]
    action: Callable[[dict], dict]  # returns dict of new/updated facts
    priority: int = 0


class SymbolicEngine:
    def __init__(self):
        self.facts: dict = {}
        self.rules: list[Rule] = []
        self.fired_log: list[str] = []

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def set_facts(self, facts: dict) -> None:
        self.facts = dict(facts)

    def update_facts(self, updates: dict) -> None:
        self.facts.update(updates)

    def forward_chain(self, max_iterations: int = 50) -> list[str]:
        """Standard recognize-act cycle: each iteration, fire ONLY the
        single highest-priority rule whose condition is satisfied and
        whose action would actually change a fact, then re-evaluate from
        scratch. Firing just one rule per round (not all matching rules)
        is deliberate - it's what makes priority ordering meaningful; if
        we fired every matching rule per round, two contradictory rules
        of different priority could both fire in the same round and the
        lower-priority one would win by executing last.

        Also detects oscillation: if the fact-base returns to a state
        we've already visited, two-or-more rules are contradicting each
        other and would loop forever - we stop and log a warning rather
        than silently truncating at max_iterations with a result that
        depends on parity of the cycle length.
        """
        self.fired_log = []
        seen_states: set[tuple] = set()

        for _ in range(max_iterations):
            state_key = tuple(sorted(self.facts.items()))
            if state_key in seen_states:
                self.fired_log.append(
                    "WARNING: oscillation detected (contradictory rules) - halting forward chain"
                )
                break
            seen_states.add(state_key)

            fired_rule = None
            for rule in self.rules:
                try:
                    if rule.condition(self.facts):
                        new_facts = rule.action(self.facts)
                        changed = any(self.facts.get(k) != v for k, v in new_facts.items())
                        if changed:
                            self.facts.update(new_facts)
                            self.fired_log.append(rule.name)
                            fired_rule = rule.name
                            break  # conflict resolution: only the top-priority match fires this round
                except Exception as e:
                    # A single bad rule must not crash the whole reasoning
                    # pipeline mid-mission - log and continue to the next rule.
                    self.fired_log.append(f"{rule.name}:ERROR:{e}")

            if fired_rule is None:
                break  # fixed point: nothing left to fire
        return self.fired_log