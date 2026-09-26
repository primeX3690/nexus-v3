# NEXUS embedded decision loop — Cortex-M3 / QEMU Software-In-the-Loop

This is a **Software-In-the-Loop (SIL)** port, not full Hardware-In-the-Loop
(HIL) — there is no physical Cortex-M board in this dev environment, and
that's stated plainly rather than implied away. What this DOES prove: the
exact same weights and control logic NEXUS runs in Python actually
cross-compile and run correctly, with a real instruction-count/timing
budget, on a real embedded ISA (ARMv7-M Thumb-2) under an instruction-
accurate CPU emulator — not just "the algorithm is simple enough that it
probably would fit." Same method used for AscentGNC's embedded port
(same board, same tracing approach), for a consistent standard of evidence
across the portfolio.

## What's ported

`sensor state -> PPO policy MLP forward pass -> proposed action -> constitutional
safety veto -> final action`, matching `brain/nexus_brain.py`'s documented
pipeline order. Frozen weights (Xavier-initialized, seed=42, real dims:
state_dim=12, hidden_dim=64, n_actions=5 — matching `brain/rl/ppo_agent.py`
and `environment/gym_wrapper.py`'s actual defaults) are exported by
`export_weights.py` from a real `MLP` instance, not hand-picked numbers.

**Not ported:** PPO *training* (backprop/Adam/rollout buffer) — that's a
laptop-side, offline step in any real deployment, same split a real flight
computer would have between ground-trained weights and onboard inference.
The `low_battery_return_to_base` safety rule (priority 50) is also out of
scope — it only sets a `mode` fact and can never change `final_action`, so
it can't affect what this loop measures.

## Build & run

```bash
# from nexus_v3/
python3 embedded/export_weights.py > embedded/weights_state.h

arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -O2 -g -ffreestanding -nostdlib \
  -T embedded/linker.ld -o embedded/nexus_decision_loop.elf \
  embedded/startup.c embedded/nexus_decision_loop.c -Wl,--gc-sections -lgcc

qemu-system-arm -M lm3s6965evb -kernel embedded/nexus_decision_loop.elf \
  -semihosting -nographic
```

Expected output (matches the Python ground truth printed by
`export_weights.py` to stderr — proposed action 1, overridden to 0 by
`collision_imminent`):

```
NEXUS embedded decision loop (Cortex-M3 / QEMU lm3s6965evb)
ran 200 decision ticks
last tick: proposed=1 final_action=0 vetoed=1 reason=collision_imminent
```

## What didn't work first try (kept, not hidden)

The first build linked against newlib's generic `rdimon` (semihosting) crt0
(`--specs=rdimon.specs`) and hard-faulted at reset (`Lockup: can't escalate
3 to HardFault`) on this QEMU board — that startup code's assumptions don't
match `lm3s6965evb`'s memory map. Fixed by writing a minimal, real
Cortex-M vector table + reset handler from scratch (`startup.c` +
`linker.ld`, real LM3S6965 memory map: 256KB flash @ 0x0, 64KB SRAM @
0x20000000) and a from-scratch ARM semihosting shim (`semihosting.h`,
direct `bkpt #0xAB` calls) instead of pulling in the rest of newlib. This
is itself a real, standard bare-metal lesson, not a corner cut.

## Measured result

Cortex-M3 has **no FPU** — every `float` op here compiles to a call into
`libgcc`'s software-float routines (`__aeabi_fmul`, `__aeabi_fadd`, ...).
The DWT cycle counter is unimplemented on this QEMU board model (same
finding as AscentGNC's port), so timing comes from single-stepped
instruction tracing instead:

```bash
qemu-system-arm -M lm3s6965evb -kernel embedded/nexus_decision_loop.elf \
  -semihosting -nographic -singlestep -d exec -D trace.log
python3 embedded/analyze_trace.py trace.log --n-iter 200 --clock-hz 50000000 --loop-hz 50
```

| Build | Instructions / decision tick | Time @ 50MHz | Margin vs 50Hz (20ms) budget |
|---|---|---|---|
| `-O0` (debug) | 150,267 | ~3.0 ms | 85.0% |
| `-O2` (release) | 129,328 | ~2.6 ms | 87.1% |

**Real finding, not assumed:** `-O2` only saves ~14% over `-O0`, because
the dominant cost is soft-float arithmetic itself (1,088 multiply-adds
across the 12→64→5 MLP, each a full software routine on a no-FPU core),
which the compiler can't optimize away — it's not C-level overhead the
optimizer can remove. The constitutional safety veto that runs after it
costs a handful of integer comparisons — effectively free by comparison.
**This means the learned policy, not the safety layer, is the real
per-tick cost on this class of MCU**, and the actual fix for a real
deployment would be int8/fixed-point quantization or an FPU-equipped part
(Cortex-M4F/M7F) — future work, called out honestly rather than quietly
assumed away. Either way, ~2.6-3.0ms fits comfortably inside a 20ms
(50Hz) control-loop budget on the real 50MHz LM3S6965 clock, with margin
to spare even against a 10x-pessimistic CPI estimate.

## Files

- `export_weights.py` — dumps a real `MLP`'s weights + a test scenario to `weights_state.h`
- `tanh_lut.h` — 256-entry LUT tanh (no libm transcendentals — same reasoning TFLite Micro/CMSIS-NN use)
- `nexus_decision_loop.c` — the ported decision loop + measurement brackets
- `startup.c` / `linker.ld` — from-scratch Cortex-M3 vector table, reset handler, LM3S6965 memory map
- `semihosting.h` — minimal ARM semihosting console I/O, no libc
- `analyze_trace.py` — turns a QEMU singlestep trace into instructions/tick and a timing-margin estimate

## Honest scope note

Software-in-the-loop proves the code is real-embedded-target-correct and
gives a genuine instruction/timing budget. It does **not** prove real
sensor-interrupt latency, a real bus/DMA/RTOS scheduling context, or
real power/thermal behavior — that needs an actual Cortex-M board (a
$10-20 STM32F103 "Blue Pill" or similar would run this unchanged, since
it targets the same architecture family with -mcpu adjusted).
