"""The parallel scoring primitive: order preservation, in-process fallback, and equality with serial."""

from __future__ import annotations

import random

import pytest

from quiver import Responds, Strain, parts
from quiver.intent import Intent
from quiver.objective import score_once
from quiver.report import acceptance
from quiver.research.objective import Context, score_under_context
from quiver.research.parallel import ScoreCell, default_workers, pmap, score_cells

_ROWS = [
    "id,name,role,regulator,detects,family,organism,ymax,ymin,K,n,quality",
    "p1,PhlF,repressor,PhlF,,TetR,Eco,3.9,0.01,0.03,4.0,measured",
    "s1,AraC_sensor,sensor,AraC,arabinose,sensor,Eco,2.5,0.008,,,measured",
    "r1,GFP,reporter,GFP,,reporter,Eco,1.0,0.001,,,measured",
]


def _intents(tmp_path) -> list[Intent]:
    path = tmp_path / "inventory.csv"
    path.write_text("\n".join(_ROWS) + "\n")
    inventory = parts(path, strain=Strain("MG1655"))
    gfp, ara = inventory.reporter("GFP"), inventory.inducer("arabinose")
    return [Intent(reports=gfp, senses=(ara,), responds=d) for d in (Responds.UP, Responds.DOWN)]


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
def test_score_cells_equals_serial_scoring(tmp_path) -> None:
    pytest.importorskip("loica")
    from quiver.generate import random_sensor

    rng = random.Random(0)
    grns = [random_sensor(rng, max_regulators=4) for _ in range(3)]
    contexts: list[Context | None] = [None, Context("host-weak", alpha_mult=0.3)]

    cells: list[ScoreCell] = []
    expected: list[float] = []
    for intent in _intents(tmp_path):
        behavior = acceptance(intent)
        for grn in grns:
            for ctx in contexts:
                cells.append(ScoreCell(grn=grn, intent=intent, context=ctx))
                expected.append(score_once(grn, behavior) if ctx is None else score_under_context(grn, behavior, ctx))

    assert score_cells(cells, workers=4) == expected
