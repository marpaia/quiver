"""Grounding the objective in a *measured* reference dose-response.

The synthetic objective (`target.py`) matches hand-set features (EC50, dynamic range). This matches a
whole *measured* transfer function instead: a `ReferenceCurve` of endpoint-vs-dose, and `curve_match`,
which scores a simulated dose-response by its log-space, shape-normalized RMSE to the reference. The
reference can come from the same simulator (for testing) or, via `reference_from_measurements`, from real
characterization data in the ``flapjack-data`` model, the same data model a wet-lab readout populates, so
grounding and validation share one schema.

The module is duck-typed (a "measurement" is anything with ``sample_id``, ``value``, ``time``), so it does
not import ``flapjack_data`` and stays fully unit-testable; the real class is exercised by a marked test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Protocol

from quiver.objective.behavior import Behavior
from quiver.objective.simulate import SimulationResult


@dataclass(frozen=True)
class ReferenceCurve:
    """A measured (or reference) dose-response: reporter endpoint per dose, for one signal."""

    signal: str
    doses: list[float]
    values: list[float]


def result_to_reference(result: SimulationResult, signal: str) -> ReferenceCurve:
    """Express a simulation's dose-response as a `ReferenceCurve` (endpoints per dose)."""
    return ReferenceCurve(signal=signal, doses=list(result.doses), values=result.endpoints(signal))


def curve_match(reference: ReferenceCurve, *, floor: float = 1e-6) -> Behavior:
    """Score a candidate by closeness of its dose-response to ``reference``: ``exp(-RMSE)`` in log space,
    mean-centered so the match is on the transfer-function *shape*, not the absolute reporter scale
    (simulated units and measured RFU differ). The reference is interpolated onto the candidate's dose
    grid in log-dose."""

    def detect(result: SimulationResult) -> float:
        endpoints = result.endpoints(reference.signal)
        if len(endpoints) < 2 or len(reference.values) < 2:
            return 0.0
        aligned = _align(reference.doses, reference.values, result.doses)
        sim = _centered_log(endpoints, floor)
        ref = _centered_log(aligned, floor)
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(sim, ref)) / len(sim))
        return math.exp(-rmse)

    return detect


def _centered_log(values: list[float], floor: float) -> list[float]:
    logs = [math.log10(max(v, floor)) for v in values]
    center = mean(logs)
    return [v - center for v in logs]


def _align(ref_doses: list[float], ref_values: list[float], target_doses: list[float]) -> list[float]:
    """Interpolate the reference curve onto ``target_doses`` (linear in log10-dose; the zero-dose baseline
    is carried as-is). Interpolating unconditionally is exact when the two dose grids coincide, and correct
    (rather than silently misaligned) when they share a length but differ in dose."""
    positive = [(d, v) for d, v in zip(ref_doses, ref_values) if d > 0.0]
    baseline = next((v for d, v in zip(ref_doses, ref_values) if d <= 0.0), ref_values[0])
    out: list[float] = []
    for dose in target_doses:
        if dose <= 0.0 or not positive:
            out.append(baseline)
        else:
            out.append(_interp_log(positive, dose))
    return out


def _interp_log(points: list[tuple[float, float]], dose: float) -> float:
    x = math.log10(dose)
    xs = [math.log10(d) for d, _ in points]
    ys = [v for _, v in points]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    pts = list(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1 and x1 != x0:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class _Measurement(Protocol):
    sample_id: int
    value: float
    time: float


def reference_from_measurements(
    measurements: Iterable[_Measurement], dose_by_sample: Mapping[int, float], signal: str
) -> ReferenceCurve:
    """Build a `ReferenceCurve` from measurements of one signal: per sample (dose), the endpoint is the
    value at the latest time; ``dose_by_sample`` maps each sample id to its inducer dose. Duck-typed, so
    it accepts ``flapjack_data.Measurement`` (``sample_id`` / ``value`` / ``time``) directly."""
    latest: dict[int, tuple[float, float]] = {}  # sample_id -> (time, value)
    for m in measurements:
        if m.sample_id not in dose_by_sample:
            continue
        if m.sample_id not in latest or m.time > latest[m.sample_id][0]:
            latest[m.sample_id] = (m.time, m.value)
    pairs = sorted((dose_by_sample[sid], value) for sid, (_, value) in latest.items())
    return ReferenceCurve(signal=signal, doses=[d for d, _ in pairs], values=[v for _, v in pairs])
