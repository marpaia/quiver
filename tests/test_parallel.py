"""The parallel scoring primitive: order preservation, in-process fallback, and equality with serial."""

from __future__ import annotations

import random

import pytest

from quiver.objective import dose_response_match, score_once, sensor_target_manifold
from quiver.research.objective import Context, score_under_context
from quiver.research.parallel import SensorCell, default_workers, pmap, score_cells


def _square(x: int) -> int:
    return x * x


def test_pmap_preserves_order_and_matches_serial() -> None:
    items = list(range(20))
    assert pmap(_square, items, workers=4) == [x * x for x in items]


def test_pmap_empty_and_single_run_in_process() -> None:
    assert pmap(_square, []) == []
    assert pmap(_square, [5], workers=8) == [25]


def test_default_workers_leaves_headroom() -> None:
    assert default_workers() >= 1


@pytest.mark.sim
def test_score_cells_equals_serial_scoring() -> None:
    pytest.importorskip("loica")
    from quiver.generate import random_sensor

    targets = sensor_target_manifold(3, seed=0)
    rng = random.Random(0)
    grns = [random_sensor(rng, max_regulators=4) for _ in range(3)]
    contexts: list[Context | None] = [None, Context("host-weak", alpha_mult=0.3)]

    cells: list[SensorCell] = []
    expected: list[float] = []
    for target in targets:
        behavior = dose_response_match(target)
        for grn in grns:
            for ctx in contexts:
                cells.append(SensorCell(grn=grn, target=target, context=ctx))
                expected.append(score_once(grn, behavior) if ctx is None else score_under_context(grn, behavior, ctx))

    assert score_cells(cells, workers=4) == expected
