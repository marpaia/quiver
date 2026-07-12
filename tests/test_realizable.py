from __future__ import annotations

from dataclasses import replace

from quiver.generate import example_inducible, is_realizable
from quiver.generate.realizable import HILL_N_RANGE, MAX_DYNAMIC_RANGE, clamp_realizable


def _set_params(grn, node_id, params):
    nodes = tuple(replace(node, params=params) if node.id == node_id else node for node in grn.nodes)
    return replace(grn, nodes=nodes)


def test_example_circuit_is_realizable() -> None:
    # The example circuit's kinetics sit inside the Cello envelope, so the clamp is a no-op.
    assert is_realizable(example_inducible())


def test_out_of_range_hill_coefficient_is_not_realizable() -> None:
    grn = example_inducible()
    beyond = HILL_N_RANGE[1] + 5.0
    unbuildable = _set_params(grn, "inv", {"alpha": [1.0, 0.0], "K": 1.0, "n": beyond})
    assert not is_realizable(unbuildable)
    # Clamping brings it back inside the envelope.
    clamped = clamp_realizable("hill1", unbuildable.node("inv").params)
    assert clamped["n"] == HILL_N_RANGE[1]


def test_excessive_dynamic_range_is_not_realizable() -> None:
    grn = example_inducible()
    huge = MAX_DYNAMIC_RANGE * 100.0
    unbuildable = _set_params(grn, "inv", {"alpha": [1.0, huge], "K": 1.0, "n": 2.0})
    assert not is_realizable(unbuildable)
