
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from utils.seeding import set_global_seed
from environment.gym_wrapper import NavigationEnv
from interface.hardware_abstraction import LightweightSimBackend, ROS2Backend, RobotBackend
from interface.mavlink_bridge import MAVLinkBridge


def test_sim_backend_implements_full_contract():
    set_global_seed(0)
    env = NavigationEnv(world_size=15, n_obstacles=2, max_steps=30, seed=0)
    backend = LightweightSimBackend(env)

    obs = backend.reset()
    assert obs.shape == env.observation_space.shape
    assert backend.is_connected()

    backend.send_action(4)
    obs2 = backend.get_observation()
    assert obs2.shape == env.observation_space.shape


def test_generic_code_works_through_the_abstraction():
    """The whole point of RobotBackend: code written against the
    interface shouldn't care which concrete backend it's running on."""
    set_global_seed(0)
    env = NavigationEnv(world_size=15, n_obstacles=1, max_steps=30, seed=0)
    backend: RobotBackend = LightweightSimBackend(env)

    def run_fixed_policy(backend: RobotBackend, n_steps: int) -> int:
        backend.reset()
        for _ in range(n_steps):
            backend.send_action(0)
        return n_steps

    assert run_fixed_policy(backend, 7) == 7


def test_ros2_backend_fails_honestly_not_silently():
    """ROS2Backend must never pretend to work - rclpy isn't installed
    here (or reliably on the target laptop without a full ROS2 system
    install), so every entry point should raise a clear, honest error
    rather than silently no-op or return fake data."""
    with pytest.raises(NotImplementedError):
        ROS2Backend()


def test_mavlink_bridge_real_loopback_roundtrip():
    """Genuine UDP loopback test - real sockets, real MAVLink
    encode/decode, not mocked. This mirrors testing against PX4 SITL
    before real hardware."""
    vehicle = MAVLinkBridge("udpin:127.0.0.1:14560", source_system=1)
    time.sleep(0.2)
    gcs = MAVLinkBridge("udpout:127.0.0.1:14560", source_system=255)

    try:
        gcs.send_heartbeat()
        msg = vehicle.receive_message(msg_type="HEARTBEAT", timeout=3.0)
        assert msg is not None
        assert msg.get_type() == "HEARTBEAT"

        gcs.master.target_system = 1
        gcs.master.target_component = 1
        gcs.send_velocity_command(vx=1.5, vy=-0.5, vz=0.0, yaw_rate=0.2)
        msg2 = vehicle.receive_message(msg_type="SET_POSITION_TARGET_LOCAL_NED", timeout=3.0)
        assert msg2 is not None
        assert abs(msg2.vx - 1.5) < 1e-3
        assert abs(msg2.vy - (-0.5)) < 1e-3
        assert abs(msg2.yaw_rate - 0.2) < 1e-3
    finally:
        vehicle.close()
        gcs.close()


def test_sitl_backend_handles_no_sitl_running():
    """SITLBackend must degrade gracefully (not crash) when no SITL
    instance is actually running - this IS testable without SITL
    installed, unlike actually flying it."""
    from interface.hardware_abstraction import SITLBackend

    backend = SITLBackend("udp:127.0.0.1:14599", connect_timeout=2.0)
    assert backend.is_connected() is False

    obs = backend.get_observation()
    assert obs.shape == (6,)

    backend.send_action(4)  # must not raise even though disconnected
    backend.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))