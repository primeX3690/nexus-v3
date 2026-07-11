# NEXUS v3 - Architecture

## Design philosophy

Zero-LLM autonomous robotics. Every intelligence component - perception,
world modeling, reasoning, control, coordination - is a classical or
neural algorithm implemented directly (mostly in pure NumPy/SciPy), not
a wrapper around a language model. This is a deliberate, contrarian bet:
that general-purpose autonomous agents don't need an LLM in the control
loop, and that a from-scratch stack is both more auditable (every
decision traces to inspectable code, not an opaque forward pass through
billions of parameters) and runs on CPU-only hardware.

## Data flow (single robot)

```
raw sensors (T, sensor_dim)
        |
        v
brain/perception/snn_layer.py (LIF spiking neurons)
        |  spike_rate() -> feature vector
        v
brain/rl/ppo_agent.py .act(state) -----------------> action, log_prob, value
        |                                                    |
        v                                                    v
brain/reasoning/symbolic_engine.py            safety/constitutional_ai.py
(forward-chaining rules)                      (hardcoded veto layer, priority-ordered)
        |                                                    |
        +--------------------- final_action <-----------------+
                                    |
                                    v
                          environment/*.py .step(action)
                                    |
                                    v
                        reward, next_state, done
                                    |
                                    v
              brain/rl/replay_buffer.py (GAE-lambda advantage calc)
                                    |
                                    v
                 brain/rl/ppo_agent.py .update() (clipped surrogate + value loss)
                                    |
                                    v
        brain/memory/{working,episodic,semantic}_memory.py (context + recall)
```

## Data flow (swarm, N robots)

```
simulation/swarm/swarm_manager.py.step()
   +--> per alive agent: swarm_agent.py.step() (Boids + stigmergy pull)
   +--> stigmergy_map.py.deposit() + .evaporate() (shared pheromone grid)
   +--> fault_tolerance.py.tick() per agent (decentralized heartbeat,
        no central coordinator to fail)

environment/multi_robot_env.py (gym-like wrapper, per-agent obs/reward)
        v
training/multi_agent_trainer.py
        +--> one INDEPENDENT ppo_agent.py PER robot (no shared weights,
             no central critic)
```

## Module dependency graph

```
utils/            <- no internal deps
core/              <- utils/
brain/perception/  <- utils/
brain/world_model/ <- utils/
brain/reasoning/   <- utils/
brain/rl/          <- utils/
brain/memory/      <- (numpy only)
brain/nexus_brain.py <- perception/, world_model/, reasoning/, rl/, memory/, utils/
simulation/swarm/  <- utils/
environment/       <- simulation/swarm/, utils/, gymnasium
safety/            <- brain/reasoning/symbolic_engine.py
training/          <- brain/rl/, safety/, utils/
demo/              <- everything (integration scripts)
```

## Why pure NumPy instead of PyTorch/JAX

1. Target hardware is a Ryzen 3 laptop, 8GB RAM, CPU-only.
2. Every PPO gradient is verified against numerical finite differences
   in tests/test_rl.py - hand-rolled backprop, but checked, not assumed.
3. Nothing here wraps a pretrained checkpoint - every decision traces to
   inspectable code.

## Hardware integration roadmap (interface/)

`interface/hardware_abstraction.py` defines one `RobotBackend` contract
(`reset`, `get_observation`, `send_action`, `is_connected`) that
`brain/nexus_brain.py`-style control code is written against, so the
brain doesn't need to change when the backend does:

- **`LightweightSimBackend`** (implemented, tested): wraps
  `environment/gym_wrapper.py`'s NavigationEnv. This is what every demo
  and test in this repo runs against today.
- **`interface/mavlink_bridge.py`'s `MAVLinkBridge`** (implemented,
  tested): a real MAVLink client using `pymavlink`, verified over an
  actual UDP loopback (`tests/test_interface.py`) - genuine message
  encode/send/receive/decode, not mocked. This is the same protocol
  PX4/ArduPilot flight controllers speak, so this bridge would talk to
  a real drone identically to how it talks to the test loopback or to
  PX4's SITL simulator.
- **`ROS2Backend`** (contract defined, NOT implemented): `rclpy` is not
  distributed on PyPI - it requires a full ROS2 system install (apt
  packages, environment sourcing on Ubuntu), unavailable in this dev
  environment and a heavy install even on the target laptop. Rather
  than ship untested ROS2 integration code, `ROS2Backend` honestly
  raises `NotImplementedError` at every method with an explanation.
  Implementing it is real-hardware-integration work planned for once
  ROS2 + a physical robot are available (see `docs/PITCH.md`'s "what
  funding unlocks" section).



- **Mamba SSM**: sequential-scan NumPy version of the selective-SSM
  recurrence, not the hardware-aware parallel-scan CUDA kernel from the
  original paper. Correct math, no throughput-parity claim.
- **Physics**: default backend is a lightweight NumPy point-mass
  simulator. Real PyBullet requires compiling an ~80MB C++ tree
  (confirmed slow even on a fast build machine); it's wired in as an
  *optional* backend, not a hard dependency.
- **MCTS**: discrete action space only.
- **SNN**: forward-pass-only spiking encoder, not trained end-to-end via
  surrogate-gradient backprop-through-spikes.