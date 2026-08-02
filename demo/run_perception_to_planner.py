"""
demo/run_perception_to_planner.py

Closes the loop the user identified as missing: previously, every MCTS
demo in this repo told the planner exactly where the goal/obstacle was
(ground truth). This demo instead has the "robot" SENSE the goal
direction through PerceptionPipeline (sensor array -> SNN -> calibrated
population-vector decode) and hands the PLANNER only that noisy,
imperfect estimate - matching what a real robot with real sensors would
have to work with.

Run: python -m demo.run_perception_to_planner
"""
from __future__ import annotations

import numpy as np

from utils.seeding import set_global_seed
from brain.perception.perception_pipeline import PerceptionPipeline
from brain.reasoning.planner import MCTSPlanner


class PerceivedWorldModel:
    """A minimal 'world model' for the planner that moves a 2D point by
    whatever action is chosen - the planner never sees true_goal_position
    directly, only what PerceptionPipeline decoded from simulated
    sensor readings (via the reward function's closure, see main())."""

    def reset(self):
        pass

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state + action


def main():
    set_global_seed(3)

    print("=" * 70)
    print("NEXUS v3 - Perception-to-Planner Demo")
    print("(MCTS plans against SENSED goal position, not ground truth)")
    print("=" * 70)

    true_goal_position = np.array([6.0, -3.0])  # only used to report accuracy at the end, never given to the planner
    robot_position = np.array([0.0, 0.0])

    pipeline = PerceptionPipeline(n_sensors=16, max_range=15.0)
    relative_true_goal = true_goal_position - robot_position

    print(f"\nTrue goal position (hidden from the planner): {true_goal_position}")
    estimate = pipeline.perceive(relative_true_goal, n_timesteps=30, sensor_noise_std=0.02)
    perceived_relative_goal = estimate["estimated_relative_position"]
    perceived_goal_position = robot_position + perceived_relative_goal

    print(f"SNN-perceived goal position (this is what the planner actually gets): "
          f"{perceived_goal_position.round(2)}")
    error = np.linalg.norm(perceived_goal_position - true_goal_position)
    cos_sim = np.dot(relative_true_goal / np.linalg.norm(relative_true_goal), estimate["direction"])
    print(f"Perception error: {error:.2f} units (direction cosine similarity: {cos_sim:.3f})")

    actions = [np.array([-1.0, 0.0]), np.array([1.0, 0.0]),
               np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]

    def reward_fn(state, action, next_state):
        # Reward moving toward the PERCEIVED goal - the planner has no
        # other information about where the goal actually is.
        dist_before = np.linalg.norm(state - perceived_goal_position)
        dist_after = np.linalg.norm(next_state - perceived_goal_position)
        return dist_before - dist_after

    planner = MCTSPlanner(PerceivedWorldModel(), actions, reward_fn, rollout_depth=4)

    print("\nPlanning a path using ONLY the perceived goal position...")
    position = robot_position.copy()
    path = [position.copy()]
    for step in range(10):
        best_idx, _ = planner.plan(position, n_simulations=150)
        position = position + actions[best_idx]
        path.append(position.copy())

    final_dist_to_true_goal = np.linalg.norm(position - true_goal_position)
    print(f"\nPath taken (planning against PERCEIVED goal only): "
          f"{[p.round(1).tolist() for p in path]}")
    print(f"Final distance to TRUE goal: {final_dist_to_true_goal:.2f} "
          f"(started {np.linalg.norm(robot_position - true_goal_position):.2f} away)")
    print()
    print("This demonstrates the full loop: sensor readings -> SNN spikes ->")
    print("calibrated population decode -> planner input -> action - with NO")
    print("ground-truth shortcut anywhere in the planning step itself.")
    print()
    print("Honest limitation: direction decoding is strong (cosine sim ~0.9+),")
    print("but DISTANCE/magnitude decoding from this simple population-vector")
    print("scheme is much weaker (see the perceived vs true goal position above) -")
    print("the planner still makes correct directional progress because the")
    print("reward function only needs relative direction to improve each step,")
    print("but absolute goal-position accuracy would need a better distance code")
    print("(e.g. rate-coded range cells) than this demo implements.")
    print("=" * 70)


if __name__ == "__main__":
    main()