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


def test_motif_readout_choice_reflects_structural_symmetry() -> None:
    # The toggle's two arms are asymmetric (only one is inducer-driven), so which arm the reporter reads is a
    # genuinely distinct wiring. The repressilator ring is symmetric, so reading any arm is the same wiring --
    # the readout choice shows up only as a distinct exact structure (via the started-high init), not a
    # distinct topology. This is the wiring/assignment distinction the name-free hash draws correctly.
    toggle_wirings = {mutual_repression_toggle(random.Random(s)).topology_hash() for s in range(12)}
    assert len(toggle_wirings) >= 2, "asymmetric toggle arms are distinct wirings"

    ring_wirings = {repressilator(random.Random(s)).topology_hash() for s in range(12)}
    ring_structures = {repressilator(random.Random(s)).structural_hash() for s in range(12)}
    assert len(ring_wirings) == 1, "a symmetric ring has one wiring whichever arm is read"
    assert len(ring_structures) >= 2, "readout/params still vary the exact structure"
