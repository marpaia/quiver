from __future__ import annotations

import random

from quiver.grn import is_operator
from quiver.research.baselines import mutual_repression_toggle, repressilator

_EXPECTED_INPUTS = {"source": 0, "receiver": 1, "hill1": 1, "hill2": 2}


def _assert_valid(grn: object) -> None:
    from quiver.grn import GRN

    assert isinstance(grn, GRN)
    assert any(n.kind == "reporter" and any(e.target == n.id for e in grn.edges) for n in grn.nodes)
    for op in (n for n in grn.nodes if is_operator(n.kind)):
        assert len(grn.inputs_of(op.id)) == _EXPECTED_INPUTS[op.kind]
        assert grn.outputs_of(op.id)


def test_toggle_is_valid_and_has_mutual_repression() -> None:
    grn = mutual_repression_toggle(random.Random(0))
    _assert_valid(grn)
    kinds = {n.kind for n in grn.nodes}
    assert "supplement" in kinds
    regulators = [n.id for n in grn.nodes if n.kind == "regulator"]
    assert len(regulators) >= 2  # two arms of the toggle


def test_repressilator_is_valid_and_autonomous() -> None:
    grn = repressilator(random.Random(0))
    _assert_valid(grn)
    assert sum(1 for n in grn.nodes if n.kind == "regulator") == 3  # the repression ring


def test_motif_readout_choice_yields_distinct_topologies() -> None:
    toggles = {mutual_repression_toggle(random.Random(s)).topology_hash() for s in range(12)}
    rings = {repressilator(random.Random(s)).topology_hash() for s in range(12)}
    assert len(toggles) >= 2 and len(rings) >= 2
