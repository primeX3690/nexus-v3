# NEXUS v3 - Resource Profile

System this was measured on: 4 physical cores, 8 logical, 7.34 GB RAM.

Every number below is from an actual `psutil`/`tracemalloc` measurement of this exact code running - not an estimate.

| Subsystem | Wall clock | Peak Python allocations (tracemalloc) | Process RSS delta | CPU (user+sys) |
|---|---|---|---|---|
| Mamba SSM (2000-step sequential scan, d_model=64) | 0.4247s | 2.772 MB | 3.02 MB | 0.406s |
| PPO training (hand-rolled MLP+backprop, 15 updates x 128 steps) | 2.1603s | 0.281 MB | 0.32 MB | 1.984s |
| 5-robot swarm (200 steps: boids + stigmergy + fault tolerance) | 1.2746s | 0.034 MB | 0.05 MB | 1.219s |
| Genetic evolution (population=25, up to 50 generations) | 5.757s | 1.863 MB | 1.7 MB | 5.469s |

**Note on RSS delta:** this measures memory added to the process *during* that specific subsystem's run - it does not include the baseline Python/numpy interpreter footprint (typically 30-60MB) that exists before any of this code runs.
