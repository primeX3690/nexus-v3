"""
brain/reasoning/planner.py

Monte Carlo Tree Search (MCTS) planner. Uses UCB1 for selection, expands
one node per simulation, and evaluates leaves by a random rollout through
the WorldModel's imagination (never touching the real environment during
planning - see brain/world_model/world_model.py).

Action space is assumed discrete (a fixed list of candidate action
vectors) - continuous MCTS needs progressive widening, which is out of
scope for this budget.
"""
from __future__ import annotations

import math
import numpy as np

from utils.seeding import get_rng


class MCTSNode:
    __slots__ = ("state", "parent", "action_from_parent", "children", "visits",
                 "total_value", "untried_actions")

    def __init__(self, state: np.ndarray, parent: "MCTSNode | None", action_from_parent,
                 available_actions: list):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = list(range(len(available_actions)))

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1(self, exploration_const: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.mean_value
        explore = exploration_const * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


class MCTSPlanner:
    def __init__(self, world_model, actions: list[np.ndarray], reward_fn,
                 exploration_const: float = 1.41, rollout_depth: int = 5,
                 seed_stream: str = "mcts_planner"):
        """
        world_model: object with .predict_next(state, action) -> next_state
        actions: fixed discrete list of candidate action vectors
        reward_fn: (state, action, next_state) -> float
        """
        self.world_model = world_model
        self.actions = actions
        self.reward_fn = reward_fn
        self.c = exploration_const
        self.rollout_depth = rollout_depth
        self.rng = get_rng(seed_stream)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and node.children:
            node = max(node.children.values(), key=lambda n: n.ucb1(self.c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            return node
        idx = node.untried_actions.pop(int(self.rng.integers(len(node.untried_actions))))
        action = self.actions[idx]
        next_state = self.world_model.predict_next(node.state, action)
        child = MCTSNode(next_state, parent=node, action_from_parent=idx, available_actions=self.actions)
        node.children[idx] = child
        return child

    def _rollout(self, node: MCTSNode) -> float:
        state = node.state
        total_reward = 0.0
        discount = 1.0
        for _ in range(self.rollout_depth):
            idx = int(self.rng.integers(len(self.actions)))
            action = self.actions[idx]
            next_state = self.world_model.predict_next(state, action)
            total_reward += discount * self.reward_fn(state, action, next_state)
            discount *= 0.95
            state = next_state
        return total_reward

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def plan(self, root_state: np.ndarray, n_simulations: int = 100) -> tuple[int, MCTSNode]:
        """Returns (best_action_index, root_node). Caller looks up
        `self.actions[best_action_index]` for the actual action vector."""
        root = MCTSNode(root_state, parent=None, action_from_parent=None, available_actions=self.actions)

        for _ in range(n_simulations):
            self.world_model.reset()  # imagination state must not leak across simulations
            node = self._select(root)
            if not node.is_fully_expanded():
                node = self._expand(node)
            value = self._rollout(node)
            self._backpropagate(node, value)

        if not root.children:
            # degenerate case: zero simulations or single-action space never expanded
            return int(self.rng.integers(len(self.actions))), root

        best_idx = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best_idx, root