
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


class SITLBackend(RobotBackend):
    """Talks to ArduPilot or PX4 SITL (Software-In-The-Loop) - a real
    flight-controller firmware binary running on your machine, simulated
    but running the ACTUAL production firmware, not a NEXUS-side model
    of one. This is a meaningfully different tier of validation than
    LightweightSimBackend: it proves NEXUS can drive real flight-control
    software, not just its own physics approximation.

    Unlike ROS2Backend, this class IS real, working code - it reuses
    interface/mavlink_bridge.py's MAVLinkBridge, which is verified over
    an actual UDP loopback in tests/test_interface.py. What's NOT
    verified in this sandbox is a live SITL instance on the other end
    (SITL requires compiling ArduPilot/PX4 from source - confirmed
    during development to need 770MB+ just for a shallow clone plus
    several GB more in submodules and a lengthy build, infeasible in
    this dev sandbox's disk/time budget). Connection/timeout handling
    IS tested here without a real SITL running (see
    tests/test_interface.py's test_sitl_backend_handles_no_sitl_running).

    Setup SITL yourself: see docs/SITL_SETUP.md for exact commands.
    Once SITL is running (e.g. `sim_vehicle.py -v ArduCopter` for
    ArduPilot), point this backend at it - default MAVLink SITL output
    is udp:127.0.0.1:14550.
    """

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550", connect_timeout: float = 5.0):
        from interface.mavlink_bridge import MAVLinkBridge
        self.bridge = MAVLinkBridge(connection_string)
        self._connected = self.bridge.wait_heartbeat(timeout=connect_timeout)
        self._last_obs = None

    def is_connected(self) -> bool:
        return self._connected

    def reset(self) -> np.ndarray:
        # SITL doesn't have a generic "reset episode" concept the way a
        # gym env does - a real/simulated vehicle just continues from
        # wherever it is. Returning the current state is the honest
        # behavior here, not a fabricated reset.
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        if not self._connected:
            return np.zeros(6)  # no telemetry available - caller should check is_connected() first
        msg = self.bridge.receive_message(msg_type="LOCAL_POSITION_NED", timeout=0.5)
        if msg is None:
            return self._last_obs if self._last_obs is not None else np.zeros(6)
        obs = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz])
        self._last_obs = obs
        return obs

    def send_action(self, action: int) -> None:
        # Maps NEXUS's small discrete action set onto velocity commands -
        # same mapping used by environment/multi_robot_env.py's ACTIONS,
        # kept consistent so a policy trained on the sim backend needs
        # no action-space translation to run against SITL.
        action_to_velocity = {
            0: (0.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, -1.0, 0.0),
            3: (-1.0, 0.0, 0.0), 4: (1.0, 0.0, 0.0),
        }
        vx, vy, vz = action_to_velocity.get(int(action), (0.0, 0.0, 0.0))
        if self._connected:
            self.bridge.send_velocity_command(vx, vy, vz)

    def close(self) -> None:
        self.bridge.close()


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