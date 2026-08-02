# ArduPilot/PX4 SITL Setup - Run This On Your Own Laptop

**Why this doc exists:** `interface/hardware_abstraction.py`'s `SITLBackend`
and `interface/mavlink_bridge.py`'s `MAVLinkBridge` are real, tested code -
but SITL itself (the flight-controller firmware simulator) was NOT built or
run in the sandboxed environment this repo was developed in. The ArduPilot
source alone is 770MB+ for a shallow clone, needs several more GB in
submodules (ChibiOS, mavlink, gtest...), and takes 20-40+ minutes to build
even on capable hardware - infeasible in that sandbox's disk/time budget.
This is real, valuable work - it just needs to happen on a machine with more
time and disk space than a quick dev sandbox. That's your laptop.

## Option A: ArduPilot SITL (recommended - simpler setup)

```bash
# 1. Clone (this alone is 770MB+, budget time/disk)
git clone --recursive https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# 2. Install prerequisites (Ubuntu/Debian - adjust for your distro)
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

# 3. Build + run SITL for a copter (takes a while the first time)
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map
```

This starts SITL broadcasting MAVLink on `udp:127.0.0.1:14550` by default -
exactly what `SITLBackend`'s default `connection_string` expects.

## Option B: PX4 SITL

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
make px4_sitl none_iris
```

PX4 SITL's default MAVLink UDP port also matches `SITLBackend`'s default.

## Once SITL is running

In a second terminal, from `nexus_v3/`:

```python
from interface.hardware_abstraction import SITLBackend

backend = SITLBackend("udp:127.0.0.1:14550")
print("Connected:", backend.is_connected())  # should print True now
```

If `is_connected()` is `True`, you're driving a real flight-controller
firmware binary (not a NEXUS-side approximation of one) - `demo/run_robot_brain.py`
can be pointed at this backend instead of the default `LightweightSimBackend`
by swapping which backend gets constructed (see `interface/hardware_abstraction.py`'s
`RobotBackend` contract - any code written against it works with either backend
unchanged).

## What this proves vs. what real hardware still requires

SITL proves NEXUS's decision-making integrates correctly with real flight-
control software and the real MAVLink protocol. It does NOT prove real-world
sensor noise, real actuator latency, real wind/physics, or real battery
constraints are handled - that's what actual hardware (see `docs/PITCH.md`'s
"what funding unlocks") is for.