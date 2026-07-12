"""Shared run configuration and discovery-metric helpers for the experiment scripts.

Run knobs are read from the environment with canonical defaults, so a large run is
``N_TARGETS=150 SEEDS=10 uv run python experiments/<script>.py`` with no code edit. A value of ``0`` for
``WORKERS`` selects the default (all cores but two). The sensor target grid and the discovery-metric
wrappers (:func:`portfolio`, :func:`distinct`) are shared so the single-target and amortization experiments
score portfolios identically. The GFlowNet stream builder lives in :mod:`experiments._gfn` (it needs torch);
this module stays torch-free so the search-only experiments import it without pulling in torch.
"""

from __future__ import annotations

import os

from quiver.eval import evaluate_portfolio
from quiver.objective import TargetResponse

Stream = list[tuple[str, float]]


def env_int(name: str, default: int) -> int:
    """An integer run knob from the environment, or ``default`` if unset or blank."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def workers() -> int | None:
    """Configured worker count, or ``None`` to let the parallel map choose the default."""
    n = env_int("WORKERS", 0)
    return n or None


# Scale knobs (used by the scaled decorrelation runs).
N_TARGETS = env_int("N_TARGETS", 150)
SEEDS = env_int("SEEDS", 10)
MAX_REGULATORS = env_int("MAX_REGULATORS", 12)

# Single-target and amortization knobs, shared across the learned-sampler comparisons.
EVAL_BUDGET = env_int("EVAL_BUDGET", 300)
TUNE_STEPS = env_int("TUNE_STEPS", 2)
TRAIN_STEPS = env_int("TRAIN_STEPS", 40)
TRAIN_BATCH = env_int("TRAIN_BATCH", 12)
MAX_STEPS = env_int("MAX_STEPS", 8)
BETA = env_float("BETA", 4.0)
PASS = env_float("PASS", 0.5)

# The 12-target sensor grid: EC50 in {0.1, 1, 10}, dynamic range in {10, 100}, direction in {up, down}.
SENSOR_TARGET_GRID = [
    TargetResponse("GFP", ec50=ec50, dynamic_range=fold, direction=direction)
    for ec50 in (0.1, 1.0, 10.0)
    for fold in (10.0, 100.0)
    for direction in ("up", "down")
]


def portfolio(stream: Stream, *, pass_threshold: float = PASS) -> tuple[int, int]:
    """Distinct high-scoring topologies and the count at the mid-quality cut, for a scored stream."""
    scores = [score for _, score in stream]
    report = evaluate_portfolio(scores, [h for h, _ in stream], passed=[s >= pass_threshold for s in scores])
    return report.distinct_passers, (report.modes_at(0.5) or 0)


def distinct(stream: Stream, *, pass_threshold: float = PASS) -> int:
    """Distinct high-scoring topologies in a scored stream."""
    scores = [score for _, score in stream]
    report = evaluate_portfolio(scores, [h for h, _ in stream], passed=[s >= pass_threshold for s in scores])
    return report.distinct_passers
