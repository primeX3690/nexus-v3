"""
interface/hardware_abstraction.py

A single interface that brain/nexus_brain.py talks to, regardless of
whether it's controlling a simulated robot or a real one. This is what
makes "train in simulation, deploy on real hardware later" possible
without rewriting the brain - only the backend changes.

Two backends exist right now:
  - LightweightSimBackend: wraps the already-tested numpy physics
    (environment/pybullet_env.py) - fully working, used by every demo
    and test in this repo today.
  - ROS2Backend: NOT implemented. ROS2's Python bindings (rclpy) are
    not distributed on PyPI - they require a full ROS2 system install
    (apt packages on Ubuntu, environment sourcing), which isn't
    available in this dev sandbox and is a heavy install even on the
    target laptop. Rather than write ROS2 integration code that has
    never been executed and could easily be subtly wrong, this class
    defines the CONTRACT a real implementation must satisfy and raises
    NotImplementedError with an honest explanation. See
    architecture/ARCHITECTURE.md's roadmap section.

Design note: every method here is deliberately synchronous and simple
(no async, no callbacks) so LightweightSimBackend and a future
ROS2Backend can share the exact same call pattern from nexus_brain.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class RobotBackend(ABC):
    """Contract every backend (simulated or real) must implement."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the robot/episode. Returns the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Returns the current sensor/state observation vector."""
        raise NotImplementedError

    @abstractmethod
    def send_action(self, action: int) -> None:
        """Applies a discrete action to the robot (sim step or real actuation)."""
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """For real backends: is the link to the robot alive? Sim
        backends can always return True."""
        raise NotImplementedError


class LightweightSimBackend(RobotBackend):
    """Wraps environment.gym_wrapper.NavigationEnv (already tested
    throughout this repo) behind the RobotBackend contract. This is the
    backend every demo/test in this repo actually runs against today."""

    def __init__(self, env):
        self.env = env
        self._last_obs = None

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._last_obs = obs
        return obs

    def get_observation(self) -> np.ndarray:
        if self._last_obs is None:
            return self.reset()
        return self._last_obs

    def send_action(self, action: int) -> None:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._last_step_result = (obs, reward, terminated, truncated, info)

    def last_step_result(self):
        """Sim-only convenience: returns the full (obs, reward, terminated,
        truncated, info) tuple from the most recent send_action() call,
        which a real backend wouldn't have in the same synchronous form
        (reward on real hardware usually comes from a separate estimator)."""
        return getattr(self, "_last_step_result", None)

    def is_connected(self) -> bool:
        return True


class ROS2Backend(RobotBackend):
    """NOT YET IMPLEMENTED - see module docstring.

    Planned contract once ROS2 is available (post-funding, on real
    hardware): reset() would call a service to reset simulation/odometry,
    get_observation() would read the latest message from subscribed
    topics (e.g. /odom, /imu, /camera), and send_action() would publish
    a geometry_msgs/Twist (or similar) to a velocity-command topic.
    """

    _NOT_IMPLEMENTED_MSG = (
        "ROS2Backend requires a full ROS2 installation (rclpy is not "
        "pip-installable - it needs the ROS2 system packages via apt, "
        "plus environment sourcing). This has NOT been built or tested "
        "because there is no way to verify it in this environment. "
        "The interface contract (RobotBackend) is defined and ready - "
        "implementing this class is real-hardware-integration work "
        "planned for once ROS2 + a robot are available. See "
        "architecture/ARCHITECTURE.md."
    )

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def reset(self) -> np.ndarray:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def get_observation(self) -> np.ndarray:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def send_action(self, action: int) -> None:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def is_connected(self) -> bool:
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)