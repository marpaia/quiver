"""Target dose-response matching: the graded objective for the sensor device class.

A sensor's design intent is a specific transfer function, switch at a target inducer level (EC50), with
a target dynamic range and direction. This scores how closely a simulated dose-response *matches* that
target, continuously, so the objective grades *among* valid sensors rather than merely passing them. That
matters twice over: it gives a search a gradient (a great sensor at the wrong threshold scores below one
at the right threshold), and it makes the landscape multimodal, many distinct topologies can hit the same
target, which is the regime a diversity-seeking search is built to exploit.

This is the sensor detector in the behavior-specification family (see ``behavior``): other device classes
(switch, oscillator, logic, homeostat) carry their own detectors over the same simulation result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import exp, log10

from quiver.objective.behavior import Behavior, fold_change
from quiver.objective.simulate import SimulationResult

# A finite penalty (in log10-dose "decades") for a curve with no discernible switch point.
_NO_EC50_PENALTY = 2.0


@dataclass(frozen=True)
class TargetResponse:
    """A sensor design target: the transfer function a candidate should reproduce.

    ``ec50`` is the inducer concentration at the half-maximal response, ``dynamic_range`` the target fold
    change (max/min), and ``direction`` whether the reporter rises (``"up"``), falls (``"down"``), or
    either (``"any"``) with dose.
    """

    signal: str
    ec50: float
    dynamic_range: float
    direction: str = "any"


def sensor_target_grid() -> list[TargetResponse]:
    """The canonical discrete sensor-target grid: EC50 ∈ {0.1, 1, 10} × dynamic range ∈ {10, 100} ×
    direction ∈ {up, down} = 12 targets. Centralizes the grid the paired-comparison experiments share."""
    return [
        TargetResponse("GFP", ec50=ec50, dynamic_range=fold, direction=direction)
        for ec50 in (0.1, 1.0, 10.0)
        for fold in (10.0, 100.0)
        for direction in ("up", "down")
    ]


def sensor_target_manifold(
    n: int,
    *,
    ec50_range: tuple[float, float] = (0.1, 1.5),
    fold_range: tuple[float, float] = (10.0, 100.0),
    directions: tuple[str, ...] = ("up", "down"),
    seed: int = 0,
) -> list[TargetResponse]:
    """``n`` sensor targets sampled from a continuous manifold: EC50 and dynamic range drawn log-uniformly
    over their ranges, direction drawn uniformly. The continuous manifold replaces the discrete grid for the
    scaled construct-validity study, so the decorrelation effect is measured over many targets rather than a
    handful. ``ec50_range`` defaults to the workable window (an EC50 the search can place inside the assay's
    dose sweep); targets outside it produce no passing corpus and are useless for the study."""
    rng = random.Random(seed)
    lo_ec50, hi_ec50 = log10(ec50_range[0]), log10(ec50_range[1])
    lo_fold, hi_fold = log10(fold_range[0]), log10(fold_range[1])
    targets: list[TargetResponse] = []
    for _ in range(max(0, n)):
        ec50 = 10.0 ** rng.uniform(lo_ec50, hi_ec50)
        fold = 10.0 ** rng.uniform(lo_fold, hi_fold)
        targets.append(TargetResponse("GFP", ec50=ec50, dynamic_range=fold, direction=rng.choice(directions)))
    return targets


def half_max_dose(result: SimulationResult, signal: str) -> float | None:
    """The dose at the half-maximal response (EC50), interpolated in log-dose; ``None`` if the curve has
    no half-crossing over the sweep. The zero-dose baseline anchors the response floor/ceiling but is not
    used as an interpolation endpoint (its log is undefined)."""
    doses = result.doses
    endpoints = result.endpoints(signal)
    if len(endpoints) != len(doses) or len(endpoints) < 3:
        return None
    lo, hi = min(endpoints), max(endpoints)
    if hi - lo <= 1e-12:
        return None
    half = 0.5 * (lo + hi)
    positive = [(d, e) for d, e in zip(doses, endpoints) if d > 0.0]
    for (d0, e0), (d1, e1) in zip(positive, positive[1:]):
        if (e0 - half) * (e1 - half) <= 0.0 and e1 != e0:
            frac = (half - e0) / (e1 - e0)
            return float(10.0 ** (log10(d0) + frac * (log10(d1) - log10(d0))))
    return None


def dose_response_match(
    target: TargetResponse, *, w_ec50: float = 1.0, w_fold: float = 1.0, w_dir: float = 1.0
) -> Behavior:
    """Score a candidate by closeness of its dose-response to ``target``: ``exp(-weighted distance)`` in
    ``(0, 1]``. Distance sums log-scale gaps in EC50 and dynamic range plus a direction mismatch, so the
    score is continuous and non-saturating, it keeps improving as the curve approaches the target."""

    def detect(result: SimulationResult) -> float:
        endpoints = result.endpoints(target.signal)
        if len(endpoints) < 2:
            return 0.0
        fold = fold_change(result, target.signal)
        d_fold = abs(log10(max(fold, 1e-9)) - log10(max(target.dynamic_range, 1e-9)))
        ec50 = half_max_dose(result, target.signal)
        d_ec50 = _NO_EC50_PENALTY if ec50 is None else abs(log10(ec50) - log10(max(target.ec50, 1e-12)))
        observed = "up" if endpoints[-1] >= endpoints[0] else "down"
        d_dir = 0.0 if target.direction in ("any", observed) else 1.0
        distance = w_ec50 * d_ec50 + w_fold * d_fold + w_dir * d_dir
        return float(exp(-distance))

    return detect
