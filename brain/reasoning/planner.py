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
                 "total_value", "untried_actions", "edge_reward")

    def __init__(self, state: np.ndarray, parent: "MCTSNode | None", action_from_parent,
                 available_actions: list, edge_reward: float | None = None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = list(range(len(available_actions)))
        # Reward earned by the specific transition (parent.state --action--> this
        # state) that created this node. None for the root (no incoming edge).
        # This used to be silently dropped during backpropagation - see
        # MCTSPlanner._backpropagate for why that was a real bug, not a style choice.
        self.edge_reward = edge_reward

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
        self.discount = 0.95
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
        # Compute the REAL immediate reward for this specific transition -
        # this used to be discarded entirely (see class docstring / node
        # edge_reward comment); now it's captured so backpropagation can
        # use it instead of relying solely on noisy random-rollout reward.
        edge_reward = self.reward_fn(node.state, action, next_state)
        child = MCTSNode(next_state, parent=node, action_from_parent=idx, available_actions=self.actions,
                          edge_reward=edge_reward)
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
            discount *= self.discount
            state = next_state
        return total_reward

    def _backpropagate(self, node: MCTSNode, rollout_value: float) -> None:
        """Standard MCTS backup, computing a proper Q-value at each node:
        Q(parent_state, action) = edge_reward + discount * V(child_state).
        This value is stored as the CHILD's own total_value/visits -
        that's what UCB1 compares between siblings at their parent
        during selection, so the edge_reward MUST be folded in before
        updating the node's own stats, not after (a subtle ordering bug
        found while testing: adding it after meant a node's own
        mean_value never reflected the reward of reaching it at all,
        only whatever came after - siblings became indistinguishable by
        their own edge quality, and search got WORSE with more
        simulations because it kept refining noise instead of signal)."""
        cumulative_value = rollout_value
        node_to_update = node
        while node_to_update is not None:
            if node_to_update.edge_reward is not None:
                cumulative_value = node_to_update.edge_reward + self.discount * cumulative_value
            node_to_update.visits += 1
            node_to_update.total_value += cumulative_value
            node_to_update = node_to_update.parent

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