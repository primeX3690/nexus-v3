"""
interface/mavlink_bridge.py

A working MAVLink bridge using pymavlink - unlike ROS2 (see
hardware_abstraction.py's ROS2Backend), pymavlink IS pip-installable
and testable without real hardware: it can talk over a UDP loopback to
a local endpoint, which is exactly how you'd test against PX4's SITL
(software-in-the-loop) simulator before ever touching a real flight
controller. tests/test_interface.py exercises this over a real
loopback socket, not mocks - the encode/decode/heartbeat round-trip
here is genuinely verified.

What this does NOT do: actually arm a real vehicle, fly real hardware,
or get tested against a real PX4 flight controller (no real drone
available). What IS verified: MAVLink message construction, sending,
and parsing work correctly over an actual network socket.
"""
from __future__ import annotations

import time

from pymavlink import mavutil


class MAVLinkBridge:
    """Thin wrapper around pymavlink for sending velocity commands and
    reading vehicle state. Works against SITL (simulated flight
    controller) or real hardware identically - MAVLink doesn't
    distinguish, which is exactly why this bridge is worth having: the
    same code path that's tested here against a loopback would work
    against a real PX4-powered drone."""

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550", source_system: int = 255):
        """connection_string examples:
        - 'udp:127.0.0.1:14550'  - SITL simulator or loopback test target
        - '/dev/ttyACM0'          - real flight controller over USB serial
        - 'udp:192.168.1.50:14550' - real drone over network telemetry
        """
        self.connection_string = connection_string
        self.master = mavutil.mavlink_connection(connection_string, source_system=source_system)

    def wait_heartbeat(self, timeout: float = 5.0) -> bool:
        """Blocks until a heartbeat is received from the other end, or
        timeout. This is the standard MAVLink handshake - you don't
        consider a link "connected" until you've seen at least one
        heartbeat from it."""
        msg = self.master.wait_heartbeat(timeout=timeout)
        return msg is not None

    def send_heartbeat(self) -> None:
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        """Sends a SET_POSITION_TARGET_LOCAL_NED message with only the
        velocity fields enabled (type_mask bits set to ignore position/
        acceleration), which is the standard way to send closed-loop
        velocity commands to a PX4/ArduPilot flight controller. This is
        the message NexusBrain's action output would map to on real
        hardware, via the (not-yet-tested) ROS2Backend or directly
        through this bridge."""
        # type_mask: ignore everything except vx, vy, vz, yaw_rate.
        # Bit meanings per MAVLink SET_POSITION_TARGET_LOCAL_NED spec.
        POSITION_IGNORE = 0b0000_0000_0000_0111
        ACCEL_IGNORE = 0b0000_1110_0000_0000
        YAW_IGNORE = 0b0100_0000_0000_0000
        type_mask = POSITION_IGNORE | ACCEL_IGNORE | YAW_IGNORE

        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,          # position (ignored)
            vx, vy, vz,        # velocity - the actual command
            0, 0, 0,          # acceleration (ignored)
            0, yaw_rate,       # yaw (ignored), yaw_rate (used)
        )

    def receive_message(self, msg_type: str | None = None, timeout: float = 1.0):
        """Non-blocking-ish receive with timeout. Returns the parsed
        message or None."""
        return self.master.recv_match(type=msg_type, blocking=True, timeout=timeout)

    def close(self) -> None:
        self.master.close()