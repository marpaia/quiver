"""Behavior detectors: read a target biological function out of a simulation result as a score in
``[0, 1]``. A detector is the qualitative goal made quantitative, the signal the design search optimizes.

These operate purely on a :class:`~quiver.objective.simulate.SimulationResult`, so they need no simulator
and are tested against hand-built results.
"""

from __future__ import annotations

from typing import Callable

from quiver.objective.simulate import SimulationResult

# A behavior detector scores how well a simulation realizes a target function, in [0, 1].
Behavior = Callable[[SimulationResult], float]


def fold_change(result: SimulationResult, signal: str) -> float:
    """Ratio of the largest to the smallest endpoint of ``signal`` across the dose sweep (1.0 if flat)."""
    endpoints = result.endpoints(signal)
    if not endpoints:
        return 1.0
    lo = min(endpoints)
    hi = max(endpoints)
    if lo <= 0.0:
        return hi / 1e-9 if hi > 0.0 else 1.0
    return hi / lo


def monotonicity(values: list[float], direction: str = "any") -> float:
    """Fraction of adjacent steps consistent with a monotone trend.

    ``direction`` is ``"up"`` (non-decreasing), ``"down"`` (non-increasing), or ``"any"`` (the larger of
    the two, a clean response in either direction). 1.0 is perfectly monotone.
    """
    if len(values) < 2:
        return 1.0
    steps = len(values) - 1
    up = sum(1 for a, b in zip(values, values[1:]) if b >= a - 1e-12) / steps
    down = sum(1 for a, b in zip(values, values[1:]) if b <= a + 1e-12) / steps
    if direction == "up":
        return up
    if direction == "down":
        return down
    return max(up, down)


def inducible_response(signal: str, *, min_fold: float = 5.0, direction: str = "any") -> Behavior:
    """A sensor detector: reward a clean, monotone dose-response with fold change up to ``min_fold``.

    A sensor responds to its inducer whether by activation or repression, so ``direction`` defaults to
    ``"any"``. Fold change saturates at ``min_fold`` (a stronger response is not rewarded past the target),
    weighted by how monotone the dose-response is, so a noisy non-monotone curve scores below a clean one.
    """

    def detect(result: SimulationResult) -> float:
        endpoints = result.endpoints(signal)
        fold = fold_change(result, signal)
        strength = min(1.0, max(0.0, (fold - 1.0) / (min_fold - 1.0))) if min_fold > 1.0 else 0.0
        return strength * monotonicity(endpoints, direction)

    return detect
