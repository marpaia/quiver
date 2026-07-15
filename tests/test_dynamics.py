from __future__ import annotations

from quiver.objective.simulate import SimulationResult
from quiver.research.objective import envelope_decay, is_sustained, sustained_oscillation, ultrasensitive_switch
from quiver.research.objective.dynamics import _history_gap


def _trace(trace: list[float]) -> SimulationResult:
    return SimulationResult(doses=[0.0], times=list(range(len(trace))), signals={"GFP": [trace]})


def test_envelope_distinguishes_sustained_from_damped() -> None:
    # A constant-amplitude oscillation holds its envelope; a decaying one loses it over the settled tail.
    sustained = _trace([0.0, 1.0, 2.0, 1.0] * 16)
    damped = _trace([0.0, 1.0, 2.0, 1.0] * 8 + [0.0, 2.0, 4.0, 2.0] * 4 + [0.0, 0.4, 0.8, 0.4] * 4)
    assert is_sustained(sustained, "GFP")
    assert not is_sustained(damped, "GFP")
    assert envelope_decay(sustained, "GFP") > 0.9
    assert envelope_decay(damped, "GFP") < 0.5


def _sweep(endpoints: list[float]) -> SimulationResult:
    # One timepoint per dose, so endpoints() returns exactly `endpoints`.
    doses = [float(i) for i in range(len(endpoints))]
    return SimulationResult(doses=doses, times=[0.0], signals={"GFP": [[e] for e in endpoints]})


def test_sustained_oscillation_rewards_periodicity() -> None:
    detect = sustained_oscillation("GFP")
    osc = [0.0, 1.0, 2.0, 1.0] * 8  # 32 points, clean period-4 oscillation
    oscillating = SimulationResult(doses=[0.0], times=list(range(len(osc))), signals={"GFP": [osc]})
    flat = SimulationResult(doses=[0.0], times=list(range(32)), signals={"GFP": [[5.0] * 32]})
    ramp = SimulationResult(doses=[0.0], times=list(range(32)), signals={"GFP": [[float(i) for i in range(32)]]})
    assert detect(oscillating) > 0.7  # strong autocorrelation periodicity
    assert detect(flat) == 0.0  # no variation
    assert detect(ramp) < 0.5  # a monotone ramp is not periodic


def test_history_gap_flags_two_branches() -> None:
    # Two initial states settle to opposite outputs at every dose: full history-dependence.
    two_branch = [[100.0] * 5, [0.0] * 5]
    assert _history_gap(two_branch) == 1.0
    # A gap over only part of the input sweep (bistable region) still registers, normalized by span.
    partial = [[100.0, 100.0, 100.0], [0.0, 0.0, 100.0]]
    assert _history_gap(partial) == 1.0


def test_history_gap_is_zero_without_two_stable_states() -> None:
    # A monostable circuit: every initial state washes out to the same settled output.
    assert _history_gap([[50.0] * 5, [50.0] * 5]) == 0.0
    # Fewer than two outcomes, or mismatched lengths, cannot show history-dependence.
    assert _history_gap([[100.0] * 5]) == 0.0
    assert _history_gap([[1.0, 2.0], [1.0, 2.0, 3.0]]) == 0.0
    assert _history_gap([]) == 0.0


def test_ultrasensitive_switch_prefers_a_sharp_transition() -> None:
    step = _sweep([1.0, 1.0, 1.0, 100.0, 100.0])
    gradual = _sweep([1.0, 25.0, 50.0, 75.0, 100.0])
    flat = _sweep([5.0, 5.0, 5.0])
    switch = ultrasensitive_switch("GFP")
    assert switch(step) > 0.9
    assert switch(step) > switch(gradual) > 0.0
    assert switch(flat) == 0.0
