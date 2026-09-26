"""
embedded/analyze_trace.py

Parses a QEMU `-singlestep -d exec -D trace.log` instruction trace and
counts REAL executed ARM Thumb-2 instructions between the bracket_start()
and bracket_end() marker functions in nexus_decision_loop.c, then divides
by N_ITER to get instructions/decision-tick.

Why this exists instead of reading a cycle counter: the DWT cycle counter
is unimplemented on QEMU's lm3s6965evb Cortex-M3 model (confirmed by
testing - reads return 0 and never advance), the same finding AscentGNC's
embedded SIL port hit on the same board. Single-step instruction tracing
is the honest fallback: it counts what actually executed, not an assumed
cycle-per-instruction ratio applied to a smaller number.

Usage:
    qemu-system-arm -M lm3s6965evb -kernel nexus_decision_loop.elf \
        -semihosting -nographic -singlestep -d exec -D trace.log
    python3 embedded/analyze_trace.py trace.log --n-iter 200 --clock-hz 50000000
"""
from __future__ import annotations

import argparse
import re


def count_instructions(trace_path: str) -> int:
    start_line = None
    end_line = None
    trace_line_re = re.compile(r"^Trace \d+:")
    with open(trace_path, "r", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "bracket_start" in line and trace_line_re.match(line):
            start_line = i
            break
    if start_line is None:
        raise RuntimeError("bracket_start not found in trace - was the binary built with "
                            "__attribute__((noinline)) markers and run to completion?")
    for i in range(start_line + 1, len(lines)):
        if "bracket_end" in lines[i] and trace_line_re.match(lines[i]):
            end_line = i
            break
    if end_line is None:
        raise RuntimeError("bracket_end not found after bracket_start - trace may be truncated "
                            "(increase the QEMU run timeout).")

    count = sum(1 for line in lines[start_line:end_line + 1] if trace_line_re.match(line))
    return count - 2  # exclude the two marker-function instructions themselves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_file")
    ap.add_argument("--n-iter", type=int, default=200, help="N_ITER from nexus_decision_loop.c")
    ap.add_argument("--clock-hz", type=float, default=50_000_000,
                     help="target MCU clock, e.g. 50MHz (LM3S6965 native) or 72MHz (STM32F103)")
    ap.add_argument("--loop-hz", type=float, default=50.0,
                     help="control loop rate to compute timing margin against")
    args = ap.parse_args()

    total = count_instructions(args.trace_file)
    per_tick = total / args.n_iter
    seconds_per_tick = per_tick / args.clock_hz
    budget_seconds = 1.0 / args.loop_hz
    margin_pct = 100.0 * (1.0 - seconds_per_tick / budget_seconds)

    print(f"Total instructions measured (bracketed region): {total}")
    print(f"N_ITER: {args.n_iter}")
    print(f"Instructions per decision tick: {per_tick:.1f}")
    print(f"At {args.clock_hz/1e6:.0f}MHz (~1 instr/cycle, real-world CPI will be somewhat higher "
          f"due to branches/memory access - this is a lower-bound-on-time estimate):")
    print(f"  time per decision tick: {seconds_per_tick*1e6:.1f} us")
    print(f"  budget at {args.loop_hz:.0f}Hz control loop: {budget_seconds*1e6:.1f} us")
    print(f"  margin: {margin_pct:.1f}%")


if __name__ == "__main__":
    main()
