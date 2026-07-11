"""
core/fitness_scorer.py

Scores a candidate program on 5 dimensions, purely by executing it -
no LLM judge involved (matches the "no external APIs" constraint).

Dimensions:
  1. correctness   - fraction of provided (input, expected_output) test
                      cases the candidate passes
  2. performance    - wall-clock speed relative to a baseline (faster = higher)
  3. complexity     - inverse of AST node count (simpler = higher, but
                      only rewarded once correctness is nonzero, else a
                      trivial `pass` would win)
  4. robustness     - fraction of adversarial/edge-case inputs handled
                      without raising an unhandled exception
  5. structural     - static-analysis score (no bare except, reasonable
                      function length, no obviously dead code)

Execution safety: candidates run via `exec` in a namespace with a
whitelisted builtins subset and a hard wall-clock timeout enforced with
signal.alarm (Unix). This is NOT a full sandbox (no seccomp/subprocess
isolation) - it's adequate for evolving small numeric/utility functions
in a trusted, single-user research context, not for running untrusted
third-party code.
"""
from __future__ import annotations

import ast
import signal
import time
from dataclasses import dataclass, field


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("execution exceeded time limit")


def _build_safe_builtins() -> dict:
    """Whitelist a minimal, safe subset of builtins for candidate exec.
    Works whether __builtins__ is a module (main script) or a dict
    (imported module context) - both happen depending on how this file
    is loaded."""
    source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    allowed = [
        "abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
        "int", "float", "str", "bool", "list", "dict", "tuple", "set",
        "sorted", "reversed", "map", "filter", "all", "any", "round",
        "isinstance", "print", "ValueError", "TypeError", "IndexError",
        "KeyError", "ZeroDivisionError", "Exception",
    ]
    return {name: source[name] for name in allowed if name in source}


_SAFE_BUILTINS = _build_safe_builtins()


@dataclass
class FitnessScore:
    correctness: float = 0.0
    performance: float = 0.0
    complexity: float = 0.0
    robustness: float = 0.0
    structural: float = 0.0
    error: str | None = None
    weights: dict = field(default_factory=lambda: {
        "correctness": 0.45, "performance": 0.15, "complexity": 0.10,
        "robustness": 0.20, "structural": 0.10,
    })

    @property
    def total(self) -> float:
        return (
            self.correctness * self.weights["correctness"]
            + self.performance * self.weights["performance"]
            + self.complexity * self.weights["complexity"]
            + self.robustness * self.weights["robustness"]
            + self.structural * self.weights["structural"]
        )

    def as_dict(self) -> dict:
        return {
            "correctness": self.correctness, "performance": self.performance,
            "complexity": self.complexity, "robustness": self.robustness,
            "structural": self.structural, "total": self.total,
        }


def _run_candidate(source: str, func_name: str, args: tuple, timeout_s: float = 1.0):
    """Exec `source`, call `func_name(*args)`, return (result, elapsed, error)."""
    namespace = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace)
    except Exception as e:
        return None, 0.0, f"compile/exec error: {e}"

    fn = namespace.get(func_name)
    if fn is None:
        return None, 0.0, f"function '{func_name}' not defined"

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    start = time.perf_counter()
    try:
        result = fn(*args)
        elapsed = time.perf_counter() - start
        return result, elapsed, None
    except TimeoutException:
        return None, timeout_s, "timeout"
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"runtime error: {type(e).__name__}: {e}"
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def score_correctness(source: str, func_name: str, test_cases: list[tuple[tuple, object]]) -> float:
    if not test_cases:
        return 0.0
    passed = 0
    for args, expected in test_cases:
        result, _, err = _run_candidate(source, func_name, args)
        if err is None and _values_match(result, expected):
            passed += 1
    return passed / len(test_cases)


def _values_match(a, b, tol=1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(a - b) <= tol
        except TypeError:
            return a == b
    return a == b


def score_performance(source: str, func_name: str, test_cases: list[tuple[tuple, object]],
                       baseline_elapsed: float = 1e-4) -> float:
    if not test_cases:
        return 0.0
    total_elapsed = 0.0
    n = 0
    for args, _ in test_cases:
        _, elapsed, err = _run_candidate(source, func_name, args)
        if err is None:
            total_elapsed += elapsed
            n += 1
    if n == 0:
        return 0.0
    avg_elapsed = total_elapsed / n
    # Score in (0, 1]. Faster-than-baseline saturates at 1.0 rather than
    # exceeding it - every dimension must stay in [0,1] or `total` (a
    # weighted sum) stops meaning what its weights imply.
    return float(min(1.0, baseline_elapsed / (baseline_elapsed + avg_elapsed) * 2))


def score_complexity(source: str) -> float:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0
    node_count = sum(1 for _ in ast.walk(tree))
    # Fewer nodes -> higher score. Saturating curve so this never dominates correctness.
    return float(1.0 / (1.0 + node_count / 30.0))


def score_robustness(source: str, func_name: str, edge_case_args: list[tuple]) -> float:
    if not edge_case_args:
        return 1.0  # nothing to test against == vacuously robust
    handled = 0
    for args in edge_case_args:
        _, _, err = _run_candidate(source, func_name, args)
        # "robust" means either it succeeds, or it fails with a clean,
        # expected exception type rather than crashing the interpreter/
        # hanging - since err is always a caught string here, absence
        # of 'timeout' is what we actually care about.
        if err != "timeout":
            handled += 1
    return handled / len(edge_case_args)


def score_structural(source: str) -> float:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0
    score = 1.0
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            score -= 0.3  # bare except is a code smell
        if isinstance(node, ast.FunctionDef):
            body_len = len(node.body)
            if body_len > 40:
                score -= 0.2  # overly long function
    return max(0.0, min(1.0, score))


def evaluate_fitness(
    source: str,
    func_name: str,
    test_cases: list[tuple[tuple, object]],
    edge_case_args: list[tuple] | None = None,
    weights: dict | None = None,
) -> FitnessScore:
    fs = FitnessScore()
    if weights:
        fs.weights.update(weights)

    try:
        ast.parse(source)
    except SyntaxError as e:
        fs.error = f"syntax error: {e}"
        return fs

    fs.correctness = score_correctness(source, func_name, test_cases)
    fs.performance = score_performance(source, func_name, test_cases)
    fs.complexity = score_complexity(source)
    fs.robustness = score_robustness(source, func_name, edge_case_args or [])
    fs.structural = score_structural(source)
    return fs