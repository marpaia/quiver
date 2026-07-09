from __future__ import annotations

import random

from quiver.generate import diverse_topologies, random_sensor, sensor_cascade, sensor_topologies, structural_mutate
from quiver.grn import is_operator

_EXPECTED_INPUTS = {"source": 0, "receiver": 1, "hill1": 1, "hill2": 2}


def test_sensor_cascade_is_valid_and_depth_sets_operator_count() -> None:
    rng = random.Random(0)
    for depth in (1, 2, 3):
        cascade = sensor_cascade(depth, rng)
        operators = [n for n in cascade.nodes if is_operator(n.kind)]
        assert len(operators) == depth
        kinds = {n.kind for n in cascade.nodes}
        assert "supplement" in kinds and "reporter" in kinds
        # Every operator's declared inputs and outputs are wired.
        for op in operators:
            assert cascade.outputs_of(op.id)


def test_deeper_cascade_is_a_distinct_topology() -> None:
    rng = random.Random(1)
    assert sensor_cascade(1, rng).topology_hash() != sensor_cascade(3, rng).topology_hash()


def test_sensor_topologies_span_multiple_topologies() -> None:
    population = sensor_topologies(40, max_depth=4, seed=0)
    topologies = {grn.topology_hash() for grn in population}
    assert len(topologies) >= 3


def test_diverse_topologies_are_valid_and_non_enumerable() -> None:
    population = diverse_topologies(80, seed=0)
    for grn in population:
        kinds = {n.kind for n in grn.nodes}
        assert "supplement" in kinds and "reporter" in kinds
        for op in (n for n in grn.nodes if is_operator(n.kind)):
            assert len(grn.inputs_of(op.id)) == _EXPECTED_INPUTS[op.kind]
            assert grn.outputs_of(op.id)
    # The enriched generator spans far more distinct topologies than the depth-only cascade family.
    assert len({grn.topology_hash() for grn in population}) >= 15


def test_structural_mutate_always_yields_a_valid_sensor() -> None:
    rng = random.Random(0)
    grn = random_sensor(random.Random(3))
    for _ in range(50):
        grn = structural_mutate(grn, rng)
        kinds = {n.kind for n in grn.nodes}
        assert "supplement" in kinds and "reporter" in kinds
        assert any(e.target == n.id for n in grn.nodes if n.kind == "reporter" for e in grn.edges)
        for op in (n for n in grn.nodes if is_operator(n.kind)):
            assert len(grn.inputs_of(op.id)) == _EXPECTED_INPUTS[op.kind]
            assert grn.outputs_of(op.id)
