from __future__ import annotations

from quiver.objective import (
    TargetResponse,
    dose_response_match,
    half_max_dose,
    sensor_target_grid,
    sensor_target_manifold,
)
from quiver.objective.simulate import SimulationResult


def _rising() -> SimulationResult:
    return SimulationResult(
        doses=[0.0, 1.0, 10.0, 100.0],
        times=[0.0],
        signals={"GFP": [[1.0], [10.0], [50.0], [100.0]]},
    )


def test_half_max_dose_falls_between_bracketing_doses() -> None:
    ec50 = half_max_dose(_rising(), "GFP")
    assert ec50 is not None and 10.0 <= ec50 <= 100.0


def test_half_max_dose_none_on_flat_curve() -> None:
    flat = SimulationResult(doses=[0.0, 1.0, 10.0], times=[0.0], signals={"GFP": [[5.0], [5.0], [5.0]]})
    assert half_max_dose(flat, "GFP") is None


def test_dose_response_match_prefers_the_closer_target() -> None:
    result = _rising()
    near = dose_response_match(TargetResponse("GFP", ec50=10.0, dynamic_range=100.0, direction="up"))(result)
    far = dose_response_match(TargetResponse("GFP", ec50=0.001, dynamic_range=2.0, direction="down"))(result)
    assert 0.0 < far < near <= 1.0


def test_dose_response_match_scores_flat_curve_low() -> None:
    flat = SimulationResult(doses=[0.0, 1.0, 10.0], times=[0.0], signals={"GFP": [[5.0], [5.0], [5.0]]})
    score = dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=10.0))(flat)
    assert score < 0.2


def test_sensor_target_grid_is_the_canonical_twelve() -> None:
    grid = sensor_target_grid()
    assert len(grid) == 12
    assert {t.ec50 for t in grid} == {0.1, 1.0, 10.0}
    assert {t.dynamic_range for t in grid} == {10.0, 100.0}
    assert {t.direction for t in grid} == {"up", "down"}


def test_sensor_target_manifold_is_deterministic_per_seed() -> None:
    a = sensor_target_manifold(50, seed=7)
    b = sensor_target_manifold(50, seed=7)
    c = sensor_target_manifold(50, seed=8)
    assert len(a) == 50
    assert [(t.ec50, t.dynamic_range, t.direction) for t in a] == [(t.ec50, t.dynamic_range, t.direction) for t in b]
    assert a != c


def test_sensor_target_manifold_samples_within_ranges() -> None:
    targets = sensor_target_manifold(200, ec50_range=(0.1, 3.0), fold_range=(10.0, 100.0), seed=1)
    assert all(0.1 <= t.ec50 <= 3.0 for t in targets)
    assert all(10.0 <= t.dynamic_range <= 100.0 for t in targets)
    assert all(t.direction in ("up", "down") for t in targets)
