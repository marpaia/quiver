"""Behavior detectors for dynamic device classes beyond the sensor: switch-like (ultrasensitive) response
and sustained oscillation, plus a two-simulation hysteresis check for bistability capacity.

The switch and oscillation detectors are :data:`~quiver.objective.behavior.Behavior` callables over a
single :class:`~quiver.objective.simulate.SimulationResult`, so they slot into the same search and
evaluation machinery as the sensor detector. Bistability proper cannot be read from one dose sweep (it is
history-dependence across initial conditions), so :func:`hysteresis` simulates from a set of opposite-biased
initial states (each regulator started dominant in turn) and is a circuit-level check, not a single-result
``Behavior``.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import mean

from quiver.grn import GRN
from quiver.objective.behavior import Behavior, fold_change
from quiver.objective.simulate import SimConfig, SimulationResult, long_run, simulate


def _autocorr_periodicity(tail: list[float]) -> float:
    """Periodicity of a settled timecourse via autocorrelation: a sustained oscillation has a strong
    secondary autocorrelation peak (after the correlation first crosses zero); a monotone or settling trace
    does not. Returns that peak height in ``[0, 1]``."""
    m = mean(tail)
    x = [v - m for v in tail]
    var = sum(v * v for v in x)
    if var < 1e-12:
        return 0.0
    zero_crossed = False
    best = 0.0
    for lag in range(1, len(x) // 2):
        a = sum(x[i] * x[i + lag] for i in range(len(x) - lag)) / var
        if a < 0.0:
            zero_crossed = True
        elif zero_crossed:
            best = max(best, a)
    return max(0.0, min(1.0, best))


def sustained_oscillation(signal: str, *, min_amplitude: float = 0.05) -> Behavior:
    """Reward a sustained oscillation in a reporter timecourse, scored by the autocorrelation periodicity
    of its settled second half, gated by relative amplitude. Read the trace under a long, constant-growth
    window (:func:`~quiver.objective.simulate.long_run`) so a limit cycle has time to establish; under the
    short saturating sensor window, oscillation cannot appear. A finite window still cannot perfectly
    separate a sustained limit cycle from a very slowly damped one; the autocorrelation peak is the
    observable signature of persistence.
    """

    def detect(result: SimulationResult) -> float:
        best = 0.0
        for trace in result.signals.get(signal, []):
            if len(trace) < 16:
                continue
            tail = trace[len(trace) // 2 :]
            low, high = min(tail), max(tail)
            if (high - low) / (abs(high) + 1e-9) < min_amplitude:
                continue
            best = max(best, _autocorr_periodicity(tail))
        return best

    return detect


def _envelope_ratio(tail: list[float]) -> float:
    """Ratio of the late-window to early-window peak-to-peak amplitude of a settled trace: near ``1`` when
    the oscillation amplitude holds (sustained), near ``0`` when it decays (damped). ``0`` when there is no
    early-window amplitude to begin with (no oscillation to sustain); clipped to ``1`` if amplitude grows."""
    half = len(tail) // 2
    if half < 2:
        return 0.0
    early, late = tail[:half], tail[half:]
    amp_early = max(early) - min(early)
    if amp_early <= 1e-9:
        return 0.0
    return min(1.0, (max(late) - min(late)) / amp_early)


def envelope_decay(result: SimulationResult, signal: str) -> float:
    """The amplitude-envelope ratio of the most periodic settled reporter trace --- the observable that
    separates a sustained limit cycle (ratio near ``1``) from a slowly-damped one (ratio well below ``1``),
    which the autocorrelation periodicity alone cannot. ``0`` if no periodic trace is present."""
    best_period, ratio = -1.0, 0.0
    for trace in result.signals.get(signal, []):
        if len(trace) < 16:
            continue
        tail = trace[len(trace) // 2 :]
        period = _autocorr_periodicity(tail)
        if period > best_period:
            best_period, ratio = period, _envelope_ratio(tail)
    return ratio


def is_sustained(result: SimulationResult, signal: str, *, min_amplitude: float = 0.05, min_ratio: float = 0.7) -> bool:
    """Whether the reporter shows a *sustained* oscillation rather than a slowly-damped one: among
    sufficient-amplitude settled traces, the most periodic one holds at least ``min_ratio`` of its amplitude
    from the early to the late half of its tail. Applied to a design that already clears the oscillation
    detector, it separates a genuine limit cycle from a decaying transient."""
    best_period, best_ratio = 0.0, 0.0
    for trace in result.signals.get(signal, []):
        if len(trace) < 16:
            continue
        tail = trace[len(trace) // 2 :]
        low, high = min(tail), max(tail)
        if (high - low) / (abs(high) + 1e-9) < min_amplitude:
            continue
        period = _autocorr_periodicity(tail)
        if period > best_period:
            best_period, best_ratio = period, _envelope_ratio(tail)
    return best_period > 0.0 and best_ratio >= min_ratio


def ultrasensitive_switch(signal: str, *, min_fold: float = 5.0) -> Behavior:
    """Reward a sharp, switch-like dose-response: a large fold change concentrated in a steep transition.

    ``sharpness`` is the fraction of the total dose-response change that falls in the single steepest
    dose step: near 1.0 for a step-like ON/OFF switch, near ``1/n`` for a gradual curve. This is
    *ultrasensitivity* read from one dose sweep, not true *bistability* (two stable states at one input);
    the latter needs a hysteresis test across initial conditions (see :func:`hysteresis`).
    """

    def detect(result: SimulationResult) -> float:
        endpoints = result.endpoints(signal)
        if len(endpoints) < 3:
            return 0.0
        total = max(endpoints) - min(endpoints)
        if total <= 0.0:
            return 0.0
        fold = fold_change(result, signal)
        strength = min(1.0, max(0.0, (fold - 1.0) / (min_fold - 1.0))) if min_fold > 1.0 else 0.0
        sharpness = max(abs(b - a) for a, b in zip(endpoints, endpoints[1:])) / total
        return strength * sharpness

    return detect


def _with_regulator_state(grn: GRN, concentrations: dict[str, float]) -> GRN:
    """A copy of ``grn`` with the named regulators' initial concentrations set. Bistability is selected by
    the *relative* starting state of the mutually-regulating species, so the two branches are reached by
    biasing regulators oppositely, not by scaling every species uniformly, which sits on the separatrix."""
    nodes = tuple(
        (
            replace(node, params={**node.params, "init_concentration": concentrations[node.id]})
            if node.id in concentrations
            else node
        )
        for node in grn.nodes
    )
    return replace(grn, nodes=nodes)


def _history_gap(outcomes: list[list[float]]) -> float:
    """The largest normalized gap between a set of settled dose-response endpoint vectors, in ``[0, 1]``.

    Each vector is the settled output reached from one initial state. A gap means two initial states settle
    to different outputs at the same input, the signature of more than one stable attractor. Pure over its
    inputs (no simulation), so the scoring logic is testable without a simulator."""
    outcomes = [o for o in outcomes if o]
    if len(outcomes) < 2 or len({len(o) for o in outcomes}) != 1:
        return 0.0
    values = [v for o in outcomes for v in o]
    span = max(max(values) - min(values), 1e-9)
    gap = max(abs(a - b) for i, o1 in enumerate(outcomes) for o2 in outcomes[i + 1 :] for a, b in zip(o1, o2))
    return min(1.0, gap / span)


def hysteresis(grn: GRN, signal: str, *, config: SimConfig | None = None, high_state: float = 100.0) -> float:
    """A bistability-capacity check: does the settled output depend on which regulator starts dominant?

    For each regulator, simulates ``grn`` from the initial state where that regulator starts high and the
    others start low, then returns the largest normalized gap between the settled dose-response endpoints
    across those states (:func:`_history_gap`). A gap means two initial states settle to different outputs
    at the same input, the two stable branches of a bistable circuit, which a single dose sweep cannot
    show. A feedforward circuit with no latching washes every initial state out to the same input-set
    output and scores zero.

    Runs one simulation per regulator, so it is a circuit-level check rather than a single-result
    ``Behavior``. Defaults to the long constant-growth window (:func:`~quiver.objective.simulate.long_run`)
    so states settle to their branches rather than freezing in the growth-saturating sensor window; two or
    more regulators are required to latch, so circuits with fewer score zero.
    """
    cfg = config or long_run()
    regulators = [node.id for node in grn.species() if node.kind == "regulator"]
    if len(regulators) < 2:
        return 0.0
    outcomes = [
        simulate(
            _with_regulator_state(grn, {r: (high_state if r == dominant else 0.0) for r in regulators}), cfg
        ).endpoints(signal)
        for dominant in regulators
    ]
    return _history_gap(outcomes)
