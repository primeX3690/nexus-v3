"""
Timing helpers. On a Ryzen 3 / CPU-only / 8GB RAM box, knowing exactly
which stage (SNN forward, MCTS rollout, PPO update...) is eating time is
essential for keeping demos runnable - hence these are used throughout
brain/ and training/ rather than added as an afterthought.
"""
from __future__ import annotations

import functools
import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """Use as a context manager or decorator:

        with Timer("mcts_rollout") as t:
            ...
        print(t.elapsed)

        @Timer("ppo_update")
        def update(...): ...
    """

    def __init__(self, label: str = "block", log_fn=print):
        self.label = label
        self.log_fn = log_fn
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        if self.log_fn is not None:
            self.log_fn(f"[Timer] {self.label}: {self.elapsed * 1000:.2f} ms")
        return False


def timed(label: str | None = None):
    """Decorator form that doesn't print by default - returns
    (result, elapsed_seconds) is NOT the behavior; instead it just times
    and logs via the module logger, keeping call signatures unchanged."""
    def decorator(func):
        name = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            wrapper.last_elapsed = elapsed
            return result

        wrapper.last_elapsed = None
        return wrapper
    return decorator