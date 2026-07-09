from __future__ import annotations

from quiver.objective import fold_change, inducible_response, monotonicity
from quiver.objective.simulate import SimulationResult


def _result() -> SimulationResult:
    # Three doses, two timepoints each; endpoints rise 1 -> 5 -> 10 across dose.
    return SimulationResult(
        doses=[0.0, 1.0, 10.0],
        times=[0.0, 1.0],
        signals={"gfp": [[0.0, 1.0], [0.0, 5.0], [0.0, 10.0]]},
    )


def test_fold_change_and_monotonicity() -> None:
    result = _result()
    assert fold_change(result, "gfp") == 10.0
    assert monotonicity(result.endpoints("gfp")) == 1.0


def test_inducible_response_saturates_and_weights_by_monotonicity() -> None:
    result = _result()
    assert inducible_response("gfp", min_fold=5.0)(result) == 1.0
    # A flat signal has no fold change, so no inducible response.
    flat = SimulationResult(doses=[0.0, 1.0], times=[0.0], signals={"gfp": [[2.0], [2.0]]})
    assert inducible_response("gfp", min_fold=5.0)(flat) == 0.0
