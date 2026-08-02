"""
benchmarks/compare_vs_llm_baseline.py

Quantitative comparison: NEXUS's actual decision-making latency/cost vs.
a typical LLM-API-based agent, on the same kind of task (pick one of N
discrete actions given a state).

METHODOLOGY - read this before trusting any number below:

  - NEXUS figures are MEASURED in this run, on this exact hardware
    (see the printed CPU info), by calling PPOAgent.act() thousands of
    times and timing it directly. Reproducible by anyone who runs this
    script.

  - LLM baseline figures are NOT measured here - this sandboxed
    environment has no network access / API keys to call a real LLM
    provider. Instead, the LLM-side numbers are drawn from PUBLICLY
    PUBLISHED figures (cited below) for typical hosted-LLM API latency
    and cost, as of early-to-mid 2025/2026 public pricing pages. They
    are explicitly ranges, not a specific benchmark run, and are
    clearly labeled as "cited, not measured" throughout the output.

  - The comparison is deliberately structural, not "NEXUS is smarter"
    - it is: NEXUS makes a decision fully offline, on a single CPU
    core, with zero marginal cost and sub-millisecond latency, because
    there is no network round-trip and no per-token billing. Whether
    an LLM-in-the-loop approach makes BETTER decisions per action is a
    genuinely separate, harder question this script does NOT claim to
    answer - only latency/cost/offline-capability are compared.

Run: python3 -m benchmarks.compare_vs_llm_baseline
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.seeding import set_global_seed
from utils.config import PPOConfig

RESULTS_DIR = Path("results")

# Publicly cited typical figures for hosted LLM API calls making a
# small-scale agentic decision (short prompt + short structured output,
# e.g. "pick an action given this state"). These are ranges reflecting
# publicly available pricing/latency reports across major providers as
# of early-to-mid 2025/2026, NOT a benchmark this script ran. Cited as
# ranges deliberately, so the comparison doesn't overstate false
# precision on the LLM side.
LLM_BASELINE_CITED = {
    "typical_latency_ms_range": (200, 2000),   # network round-trip + inference, small prompt
    "typical_cost_usd_per_call_range": (0.0001, 0.01),  # small prompt+completion, varies hugely by model tier
    "requires_network": True,
    "source_note": (
        "Publicly published typical ranges for hosted LLM API calls with "
        "short prompts, general industry knowledge as of ~2025/2026 public "
        "pricing pages - NOT measured in this environment (no network/API "
        "access here). Treat as an order-of-magnitude reference, not a "
        "precise benchmark."
    ),
}


def measure_nexus_decision_latency(n_trials: int = 2000) -> dict:
    from brain.rl.ppo_agent import PPOAgent

    set_global_seed(0)
    agent = PPOAgent(state_dim=12, n_actions=5, hidden_dim=64, config=PPOConfig())
    state = np.random.default_rng(0).normal(0, 1, 12)

    for _ in range(50):  # warm-up, avoid measuring first-call overhead
        agent.act(state)

    latencies_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        agent.act(state)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "n_trials": n_trials,
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "decisions_per_sec_single_core": float(1000 / np.mean(latencies_ms)),
        "requires_network": False,
        "marginal_cost_usd_per_decision": 0.0,  # just local CPU electricity, no per-call billing
        "cpu_info": platform.processor() or platform.machine(),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Measuring NEXUS decision latency (real, on this hardware)...")
    nexus = measure_nexus_decision_latency()
    print(f"  mean={nexus['mean_ms']:.4f}ms  p50={nexus['p50_ms']:.4f}ms  "
          f"p99={nexus['p99_ms']:.4f}ms  ({nexus['decisions_per_sec_single_core']:.0f} decisions/sec)")

    llm_mid_latency = sum(LLM_BASELINE_CITED["typical_latency_ms_range"]) / 2
    speedup = llm_mid_latency / nexus["mean_ms"]

    report = {"nexus_measured": nexus, "llm_baseline_cited": LLM_BASELINE_CITED,
              "approx_latency_speedup_factor": round(speedup, 1)}
    (RESULTS_DIR / "llm_comparison_report.json").write_text(json.dumps(report, indent=2))

    # Plot: log-scale bar chart, since the gap spans orders of magnitude
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["NEXUS\n(measured, this run)", "Typical hosted LLM API\n(cited range, not measured)"]
    low = [nexus["mean_ms"], LLM_BASELINE_CITED["typical_latency_ms_range"][0]]
    high = [nexus["p99_ms"], LLM_BASELINE_CITED["typical_latency_ms_range"][1]]
    x = np.arange(2)
    ax.bar(x, high, color=["#2a9d8f", "#e76f51"], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("decision latency (ms, log scale)")
    ax.set_title("Decision latency: NEXUS (measured) vs. typical LLM API (cited)\n"
                  "NOTE: LLM bar is a publicly-cited range, not a benchmark run here")
    for i, v in enumerate(high):
        ax.text(i, v * 1.15, f"~{v:.3g}ms", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "latency_comparison.png", dpi=120)
    plt.close()

    md = [
        "# NEXUS vs. LLM-Agent Baseline - Latency & Cost\n",
        "**Methodology note (read first):** NEXUS numbers below are measured "
        "in this exact run, on this exact hardware, by timing `PPOAgent.act()` "
        f"over {nexus['n_trials']} trials. LLM baseline numbers are NOT measured - "
        "this environment has no network/API access - and are cited public "
        "typical ranges instead, explicitly labeled as such. Do not read the "
        "LLM figures as a precise benchmark.\n",
        "| | NEXUS (measured) | Typical hosted LLM API (cited range) |",
        "|---|---|---|",
        f"| Mean decision latency | **{nexus['mean_ms']:.4f} ms** | "
        f"{LLM_BASELINE_CITED['typical_latency_ms_range'][0]}-"
        f"{LLM_BASELINE_CITED['typical_latency_ms_range'][1]} ms (cited) |",
        f"| Decisions/sec (1 CPU core) | **{nexus['decisions_per_sec_single_core']:.0f}** | "
        "~0.5-5 (network round-trip bound) |",
        f"| Marginal cost per decision | **$0** (local compute) | "
        f"${LLM_BASELINE_CITED['typical_cost_usd_per_call_range'][0]}-"
        f"{LLM_BASELINE_CITED['typical_cost_usd_per_call_range'][1]} (cited) |",
        "| Requires network | **No** | Yes |",
        "\n![latency comparison](latency_comparison.png)\n",
        f"\nApprox. latency speedup vs. cited LLM midpoint: **~{speedup:.0f}x** "
        "(structural - no network round-trip, no per-token generation - "
        "not a claim about decision QUALITY, only speed/cost/offline-capability.)\n",
        f"\n> {LLM_BASELINE_CITED['source_note']}\n",
    ]
    (RESULTS_DIR / "LLM_COMPARISON.md").write_text("\n".join(md))
    print(f"\nWrote results/LLM_COMPARISON.md, results/latency_comparison.png, "
          f"results/llm_comparison_report.json")


if __name__ == "__main__":
    main()