"""
safety/constitutional_ai.py

Hard safety constraints that sit BETWEEN the learned PPO policy and
actuation, and can only restrict/override actions, never expand what's
allowed. Built on the same SymbolicEngine used for general reasoning
(brain/reasoning/symbolic_engine.py) because these rules need the exact
same properties: auditable, deterministic, no learned weights, and
robust to a single bad rule (errors are logged, not fatal).

This is explicitly NOT an LLM constitution (no natural-language
principles interpreted by a model) - every rule here is a plain Python
predicate over a fixed set of named facts, so the full rule set can be
printed, reviewed, and unit-tested exhaustively, which is the entire
point of a safety layer for physical robots.
"""
from __future__ import annotations

from brain.reasoning.symbolic_engine import SymbolicEngine, Rule


DEFAULT_HALT_ACTION = 0  # convention: action 0 is always "stop/no-op" across environments in this project


def build_default_constitution(halt_action: int = DEFAULT_HALT_ACTION) -> SymbolicEngine:
    """Returns a SymbolicEngine pre-loaded with a baseline set of
    hardcoded robot safety rules. Callers should add domain-specific
    rules on top (e.g. via engine.add_rule(...)) rather than replacing
    these - they represent the minimum floor, not a complete policy."""
    engine = SymbolicEngine()

    engine.add_rule(Rule(
        name="critical_battery_halt",
        condition=lambda f: f.get("battery_pct", 100) < 5,
        action=lambda f: {"veto": True, "override_action": halt_action, "veto_reason": "critical_battery"},
        priority=100,
    ))

    engine.add_rule(Rule(
        name="collision_imminent_halt",
        condition=lambda f: f.get("min_obstacle_distance", 999) < 0.3,
        action=lambda f: {"veto": True, "override_action": halt_action, "veto_reason": "collision_imminent"},
        priority=95,
    ))

    engine.add_rule(Rule(
        name="out_of_bounds_halt",
        condition=lambda f: f.get("in_bounds", True) is False,
        action=lambda f: {"veto": True, "override_action": halt_action, "veto_reason": "out_of_bounds"},
        priority=90,
    ))

    engine.add_rule(Rule(
        name="low_battery_return_to_base",
        condition=lambda f: 5 <= f.get("battery_pct", 100) < 20 and f.get("mode") not in ("returning", "halted"),
        action=lambda f: {"mode": "returning", "veto_reason": "low_battery_advisory"},
        priority=50,
    ))

    engine.add_rule(Rule(
        name="communication_lost_hold_position",
        condition=lambda f: f.get("steps_since_last_comm", 0) > f.get("comm_timeout_steps", 30)
                             and f.get("mode") != "halted",
        action=lambda f: {"veto": True, "override_action": halt_action, "veto_reason": "communication_lost"},
        priority=85,
    ))

    return engine


class ConstitutionalSafetyLayer:
    """Thin, stateful wrapper: call check(proposed_action, world_facts)
    each control-loop tick to get back the (possibly overridden) action
    plus an audit trail of what fired."""

    def __init__(self, halt_action: int = DEFAULT_HALT_ACTION):
        self.engine = build_default_constitution(halt_action)
        self.veto_history: list[dict] = []

    def add_rule(self, rule: Rule) -> None:
        self.engine.add_rule(rule)

    def check(self, proposed_action: int, world_facts: dict) -> tuple[int, bool, list[str]]:
        """Returns (final_action, was_vetoed, fired_rule_names)."""
        self.engine.set_facts({**world_facts, "proposed_action": proposed_action})
        fired = self.engine.forward_chain()

        was_vetoed = bool(self.engine.facts.get("veto", False))
        final_action = int(self.engine.facts.get("override_action", proposed_action)) if was_vetoed else proposed_action

        if was_vetoed:
            self.veto_history.append({
                "proposed_action": proposed_action,
                "final_action": final_action,
                "reason": self.engine.facts.get("veto_reason", "unknown"),
                "fired_rules": fired,
            })
        return final_action, was_vetoed, fired