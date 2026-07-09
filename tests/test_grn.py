from __future__ import annotations

from dataclasses import replace

import pytest

from quiver.generate import example_inducible
from quiver.grn import GRN, Edge, default_params, is_operator


def test_example_round_trips_through_dict() -> None:
    grn = example_inducible()
    restored = GRN.from_dict(grn.to_dict())
    assert len(restored.nodes) == len(grn.nodes)
    assert restored.structural_hash() == grn.structural_hash()


def test_from_dict_rejects_unknown_kind() -> None:
    # A malformed kind fails at the deserialization boundary, not cryptically inside the simulator.
    with pytest.raises(ValueError, match="unknown node kind"):
        GRN.from_dict({"nodes": [{"id": "n0", "kind": "promoter", "name": "n0"}], "edges": []})


def test_from_dict_rejects_non_list_fields() -> None:
    with pytest.raises(ValueError, match="list-valued"):
        GRN.from_dict({"nodes": {"id": "n0"}, "edges": []})


def test_structural_hash_invariant_to_node_ordering() -> None:
    grn = example_inducible()
    shuffled = GRN(nodes=tuple(reversed(grn.nodes)), edges=grn.edges)
    assert shuffled.structural_hash() == grn.structural_hash()


def test_structural_hash_invariant_to_ids() -> None:
    grn = example_inducible()
    remap = {node.id: f"x{index}" for index, node in enumerate(grn.nodes)}
    nodes = tuple(replace(node, id=remap[node.id]) for node in grn.nodes)
    edges = tuple(Edge(remap[e.source], remap[e.target], e.port) for e in grn.edges)
    assert GRN(nodes=nodes, edges=edges).structural_hash() == grn.structural_hash()


def test_topology_hash_ignores_parameters() -> None:
    grn = example_inducible()
    node = grn.node("rec")
    tuned = replace(node, params={**node.params, "K": 9.0})
    variant = GRN(nodes=tuple(tuned if n.id == "rec" else n for n in grn.nodes), edges=grn.edges)
    # Same wiring, different Hill value: one topology, two exact structures.
    assert variant.topology_hash() == grn.topology_hash()
    assert variant.structural_hash() != grn.structural_hash()


def test_wiring_resolution() -> None:
    grn = example_inducible()
    assert grn.inputs_of("rec") == ["atc"]
    assert grn.outputs_of("rec") == ["tetr"]
    assert grn.inputs_of("inv") == ["tetr"]
    assert grn.outputs_of("inv") == ["gfp"]


def test_is_operator_and_default_params() -> None:
    assert is_operator("hill1")
    assert not is_operator("regulator")
    assert set(default_params("receiver")) == {"alpha", "K", "n"}
    # Defaults are copies: mutating one does not leak into the next.
    first = default_params("hill2")
    assert isinstance(first["alpha"], list)
    first["alpha"][0] = 99.0
    assert default_params("hill2")["alpha"] == [1.0, 0.0, 0.0, 0.0]
