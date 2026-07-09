from __future__ import annotations

from dataclasses import dataclass

import pytest

from quiver.objective import ReferenceCurve, curve_match, reference_from_measurements, result_to_reference
from quiver.objective.simulate import SimulationResult


def _result(endpoints: list[float]) -> SimulationResult:
    doses = [0.0, 1.0, 10.0, 100.0]
    return SimulationResult(doses=doses, times=[0.0], signals={"GFP": [[v] for v in endpoints]})


def test_curve_match_rewards_matching_shape() -> None:
    reference = ReferenceCurve("GFP", doses=[0.0, 1.0, 10.0, 100.0], values=[1.0, 10.0, 50.0, 100.0])
    match = curve_match(reference)
    assert match(_result([1.0, 10.0, 50.0, 100.0])) == pytest.approx(1.0)  # identical shape -> RMSE 0
    assert match(_result([5.0, 5.0, 5.0, 5.0])) < 0.7  # flat curve does not match a rising reference


def test_curve_match_is_scale_invariant() -> None:
    reference = ReferenceCurve("GFP", doses=[0.0, 1.0, 10.0, 100.0], values=[1.0, 10.0, 50.0, 100.0])
    # Same shape, 10x the absolute scale -> mean-centered log RMSE is still ~0.
    assert curve_match(reference)(_result([10.0, 100.0, 500.0, 1000.0])) == pytest.approx(1.0)


def test_result_to_reference_round_trips_endpoints() -> None:
    reference = result_to_reference(_result([2.0, 4.0, 8.0, 16.0]), "GFP")
    assert reference.doses == [0.0, 1.0, 10.0, 100.0]
    assert reference.values == [2.0, 4.0, 8.0, 16.0]


def test_reference_from_measurements_takes_latest_time_per_sample() -> None:
    @dataclass
    class M:
        sample_id: int
        value: float
        time: float

    measurements = [M(0, 1.0, 0.0), M(0, 8.0, 5.0), M(1, 2.0, 0.0), M(1, 20.0, 5.0)]
    reference = reference_from_measurements(measurements, {0: 0.0, 1: 10.0}, "GFP")
    assert reference.doses == [0.0, 10.0]
    assert reference.values == [8.0, 20.0]  # endpoint = value at the latest time


def test_reference_from_real_flapjack_measurements() -> None:
    fd = pytest.importorskip("flapjack_data")
    measurements = [
        fd.Measurement(sample_id=1, signal_id=1, value=100.0, time=1.0),
        fd.Measurement(sample_id=1, signal_id=1, value=500.0, time=5.0),
        fd.Measurement(sample_id=2, signal_id=1, value=50.0, time=5.0),
    ]
    reference = reference_from_measurements(measurements, {1: 0.0, 2: 10.0}, "GFP")
    assert reference.doses == [0.0, 10.0]
    assert reference.values == [500.0, 50.0]
