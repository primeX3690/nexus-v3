# Realism Upgrades — Status & Laptop-Side Instructions

Six specific "make it more real" upgrades were requested. Three were
buildable and verifiable inside this dev sandbox; three genuinely require
your own laptop's hardware/kernel and are documented here with exact
commands, the same honest split `docs/SITL_SETUP.md` already used for
SITL alone.

## Done in this sandbox (verified, not just written)

| # | Upgrade | Where | Verified result |
|---|---|---|---|
| 3 | Vendor-datasheet sensor noise (ICM-20948 IMU, RPLIDAR A1M8 LiDAR) instead of generic `np.random.normal` | `interface/sensor_noise_models.py`, wired into `brain/perception/sensor_array.py` | Ran the model: sub-mm noise below 1.5m, ~1% of distance beyond, real dropout beyond 12m/below 0.15m; IMU shows realistic ±5dps initial bias dominating over the 0.015dps/√Hz white-noise floor |
| 4 | Core decision loop cross-compiled for real ARM Cortex-M3, run under QEMU | `embedded/` (see `embedded/README.md`) | Built with `arm-none-eabi-gcc`, ran under `qemu-system-arm -M lm3s6965evb`; output bit-matches the Python ground truth; measured ~129K real instructions/tick (-O2) via singlestep tracing, ~2.6ms @ 50MHz, 87% margin vs a 50Hz loop |
| 6 | Swarm as real OS processes + real sockets, not one Python process | `simulation/swarm/multi_process_agent.py`, `demo/run_multi_process_swarm.py`, `tests/test_multi_process_swarm.py` | Ran it: 5 real PIDs, real SIGKILL of one, survivors detected it via real UDP-socket silence at step 14 - not a shared-memory flag flip |

## Needs your own laptop (documented, not faked)

### 1. ArduPilot/PX4 SITL

Already covered in full in `docs/SITL_SETUP.md` - not repeated here.
Short version: `interface/hardware_abstraction.py`'s `SITLBackend` and
`interface/mavlink_bridge.py` are real, tested code; SITL itself (770MB+
clone, several GB of submodules, 20-40min build) is too heavy for this
sandbox. Run `Tools/autotest/sim_vehicle.py -v ArduCopter --console --map`
on your machine, then point `SITLBackend("udp:127.0.0.1:14550")` at it.

### 2. Real webcam feed for SNN perception

This sandbox has **no camera device at all** - not a software
restriction, there's genuinely no `/dev/video*` here. This has to run on
your laptop. The perception pipeline (`brain/perception/perception_pipeline.py`)
currently takes a `(2,)` relative-position vector; a webcam feed needs a
detector stage in front of it to turn pixels into that same relative-position
signal, so the SNN itself is untouched - only its input source changes.

```bash
pip install opencv-python --break-system-packages
```

```python
# webcam_perception_demo.py — run this ON YOUR LAPTOP (needs a real camera)
import cv2
import numpy as np
from brain.perception.perception_pipeline import PerceptionPipeline  # adjust import to your setup

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("No webcam found - this genuinely needs your laptop's camera")

# Minimal color-blob "sensor": stand-in for a real object detector.
# Track the largest bright/saturated blob as the "object" NEXUS senses -
# swap this for a real detector (e.g. a lightweight OpenCV Haar/DNN
# detector) once you have a specific target class in mind.
lower = np.array([0, 120, 70]); upper = np.array([10, 255, 255])  # rough "red object" HSV range

while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            # normalize to a relative-position vector like the synthetic
            # demos use: center of frame = (0,0), scale by frame size
            rel_pos = np.array([(cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)]) * 5.0
            print("real-webcam relative position ->", rel_pos)
            # feed rel_pos into PerceptionPipeline / DirectionTunedSensorArray.sense()
            # exactly like the synthetic demos do - only the SOURCE changed.
    cv2.imshow("NEXUS webcam perception (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

What this proves that synthetic images can't: real lighting variation,
real sensor noise/motion blur, and real detector false-positives/misses -
run it in a normal room and a dim room back to back and compare how often
the blob detector drops the object, which is exactly the "real-world
noise/lighting" credibility gap synthetic data can't expose.

### 5. `tc netem` real packet-loss injection

Tried this in the sandbox first, not assumed away: installed `iproute2`
(the package providing `tc`), but the sandbox's kernel has no `sch_netem`
module available (`modprobe` doesn't even exist here, and creating a test
`dummy0` interface to attach a qdisc to was refused - this is a shared,
restricted container kernel, not a full Linux install). This needs your
own Ubuntu/Debian laptop, which ships `sch_netem` as a normal kernel
module.

```bash
# On your laptop (needs root / sudo):
sudo modprobe sch_netem   # usually already loaded, but confirm
sudo tc qdisc add dev lo root netem loss 30% delay 80ms 20ms distribution normal
```

Then re-run the swarm's real network path with that qdisc active. Two
options depending on which swarm you want to stress:

- **Single-process swarm** (`demo/run_network_chaos.py`): this one
  currently injects loss *in Python* (`simulation/swarm/network_chaos.py`),
  not over a real socket, so `tc netem` on `lo` won't touch it - it isn't
  using the network stack at all. Real OS-level chaos needs the
  multi-process swarm below instead.
- **Multi-process swarm** (`demo/run_multi_process_swarm.py`, built for
  point 6 above): this one *does* send real UDP packets over
  `127.0.0.1`, so `tc qdisc add dev lo root netem loss 30% delay 80ms`
  on your laptop will make those packets actually drop/delay at the
  kernel level before `demo/run_multi_process_swarm.py`'s
  `FaultToleranceManager` ever sees them - a genuine OS-level chaos test
  of the exact same decentralized fault-detection code already verified
  against the real-SIGKILL scenario. Remove it after with:

```bash
sudo tc qdisc del dev lo root netem
```

Compare `presumed_dead_peers` false-positive rates with netem loss at
0%/10%/30%/50% against the already-documented Python-level finding in
`demo/run_network_chaos.py` (false positives above ~50% loss with the
default timeout) - if the numbers land in the same ballpark, that's a
real cross-validation of the earlier finding from a completely different
(actual OS network stack) mechanism, not just re-confirming the same
simulated code path twice.
