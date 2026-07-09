"""Structural certificates: monotonicity (Angeli-Sontag) and feedback-loop signs (Thomas' rule)."""

from __future__ import annotations

import random

from quiver.generate.motifs import mutual_repression_toggle, repressilator
from quiver.generate.topology import sensor_cascade
from quiver.grn import GRN, Edge, Node
from quiver.objective.certificate import certifies, is_monotone, loop_signs, signed_interaction_graph


def _incoherent_feedforward() -> GRN:
    """One regulator driving the reporter through both an activating and a repressing operator: two paths
    of opposite sign reconverge, so the dose-response is non-monotone."""
    nodes = (
        Node("sup", "supplement", "aTc", {}),
        Node("r0", "regulator", "r0", {"degradation_rate": 1.0, "init_concentration": 0.0}),
        Node("recv", "receiver", "recv", {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}),
        Node("opP", "hill1", "opP", {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}),
        Node("opN", "hill1", "opN", {"alpha": [100.0, 0.0], "K": 1.0, "n": 2.0}),
        Node("out", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0}),
    )
    edges = (
        Edge("sup", "recv"),
        Edge("recv", "r0"),
        Edge("r0", "opP"),
        Edge("opP", "out"),
        Edge("r0", "opN"),
        Edge("opN", "out"),
    )
    return GRN(nodes=nodes, edges=edges)


def test_linear_cascade_is_monotone_and_acyclic() -> None:
    grn = sensor_cascade(3, random.Random(0))
    assert is_monotone(grn) is True
    assert loop_signs(grn) == (False, False)
    assert certifies(grn, "sensor") is True


def test_incoherent_feedforward_is_not_monotone() -> None:
    grn = _incoherent_feedforward()
    assert is_monotone(grn) is False
    assert certifies(grn, "sensor") is False
    assert loop_signs(grn) == (False, False)


def test_toggle_has_positive_loop() -> None:
    grn = mutual_repression_toggle(random.Random(0))
    has_positive, has_negative = loop_signs(grn)
    assert has_positive is True
    assert has_negative is False
    assert certifies(grn, "switch") is True
    assert certifies(grn, "oscillator") is False


def test_repressilator_has_negative_loop() -> None:
    grn = repressilator(random.Random(0))
    has_positive, has_negative = loop_signs(grn)
    assert has_negative is True
    assert has_positive is False
    assert certifies(grn, "oscillator") is True
    assert is_monotone(grn) is False


def test_hill2_xor_edge_is_non_monotone() -> None:
    """A two-input XOR-like logic contributes non-monotone (sign 0) edges."""
    nodes = (
        Node("a", "regulator", "a", {}),
        Node("b", "regulator", "b", {}),
        Node("xor", "hill2", "xor", {"alpha": [0.0, 100.0, 100.0, 0.0], "K": [1.0, 1.0], "n": [2.0, 2.0]}),
        Node("out", "reporter", "GFP", {"signal_id": "gfp"}),
    )
    edges = (Edge("a", "xor", 0), Edge("b", "xor", 1), Edge("xor", "out"))
    grn = GRN(nodes=nodes, edges=edges)
    signs = {(src, dst): sign for src, dst, sign in signed_interaction_graph(grn)}
    assert signs[("a", "out")] == 0
    assert signs[("b", "out")] == 0
    assert is_monotone(grn) is False
