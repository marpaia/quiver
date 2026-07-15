"""The GRN design object: a gene-regulatory network as a bipartite graph of genetic species and
regulatory operators, the unit a design search proposes, mutates, and hands to the simulator.

Nodes are either *species* (``regulator``, ``reporter``, ``supplement``) or *operators* (``source``,
``receiver``, ``hill1``, ``hill2``, ``sum``). Edges wire species into operator input ports and operators
into the species they produce. Wiring is graph-owned; a node stores only its kinetic parameters. The
model mirrors LOICA's ``GeneticNetwork`` so a design round-trips to a genetic-circuit build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Literal, cast

NodeKind = Literal["regulator", "reporter", "supplement", "source", "receiver", "hill1", "hill2", "sum"]
NodeCategory = Literal["species", "operator"]

# A scalar, vector, nested-vector, or string parameter value (Hill operators take vectors of coefficients).
ParamValue = float | int | list[float] | list[list[float]] | str

SPECIES_KINDS: frozenset[str] = frozenset({"regulator", "reporter", "supplement"})
OPERATOR_KINDS: frozenset[str] = frozenset({"source", "receiver", "hill1", "hill2", "sum"})


@dataclass(frozen=True)
class ParamSpec:
    """One editable parameter of a node kind, with its default value."""

    key: str
    default: ParamValue


@dataclass(frozen=True)
class NodeSpec:
    """The static description of a node kind: its LOICA class, role, input arity, and parameters."""

    kind: NodeKind
    category: NodeCategory
    loica_class: str
    params: tuple[ParamSpec, ...]
    # Number of input ports; ``None`` for a variable-input operator (``sum``), 0 for a source/species.
    n_inputs: int | None


NODE_SPECS: dict[NodeKind, NodeSpec] = {
    "regulator": NodeSpec(
        "regulator",
        "species",
        "Regulator",
        (ParamSpec("degradation_rate", 0.0), ParamSpec("init_concentration", 0.0)),
        0,
    ),
    "reporter": NodeSpec(
        "reporter",
        "species",
        "Reporter",
        (ParamSpec("signal_id", ""), ParamSpec("degradation_rate", 0.0), ParamSpec("init_concentration", 0.0)),
        0,
    ),
    "supplement": NodeSpec("supplement", "species", "Supplement", (), 0),
    "source": NodeSpec("source", "operator", "Source", (ParamSpec("rate", 10.0),), 0),
    "receiver": NodeSpec(
        "receiver",
        "operator",
        "Receiver",
        (ParamSpec("alpha", [0.0, 100.0]), ParamSpec("K", 1.0), ParamSpec("n", 2.0)),
        1,
    ),
    "hill1": NodeSpec(
        "hill1",
        "operator",
        "Hill1",
        (ParamSpec("alpha", [1.0, 0.0]), ParamSpec("K", 1.0), ParamSpec("n", 2.0)),
        1,
    ),
    "hill2": NodeSpec(
        "hill2",
        "operator",
        "Hill2",
        (ParamSpec("alpha", [1.0, 0.0, 0.0, 0.0]), ParamSpec("K", [1.0, 1.0]), ParamSpec("n", [2.0, 2.0])),
        2,
    ),
    "sum": NodeSpec(
        "sum",
        "operator",
        "Sum",
        (ParamSpec("alpha", [[0.0, 1.0]]), ParamSpec("K", [1.0]), ParamSpec("n", [2.0])),
        None,
    ),
}


def is_operator(kind: NodeKind) -> bool:
    return NODE_SPECS[kind].category == "operator"


def _validated_kind(kind: object) -> NodeKind:
    """Check a deserialized node kind against the registry so a malformed dict fails at the boundary
    rather than cryptically deep in the simulator."""
    if kind not in NODE_SPECS:
        raise ValueError(f"unknown node kind {kind!r}; expected one of {sorted(NODE_SPECS)}")
    return cast(NodeKind, kind)


def default_params(kind: NodeKind) -> dict[str, ParamValue]:
    """A fresh copy of the default parameters for a node kind."""
    return {spec.key: _clone(spec.default) for spec in NODE_SPECS[kind].params}


def _clone(value: ParamValue) -> ParamValue:
    if isinstance(value, list):
        return cast(ParamValue, [list(entry) if isinstance(entry, list) else entry for entry in value])
    return value


@dataclass(frozen=True)
class Node:
    """One node: a species or operator, its display name, and its kinetic parameters.

    ``component_id`` records the SBOL Component a node is grounded in (its kinetics taken from a characterized
    part); ``None`` for an abstract node. It is provenance only, carried through serialization but excluded
    from the structural and topology hashes and never passed to the simulator --- the node's identity for
    diversity and its behavior are set by its kinetics and wiring, which the component determines.
    """

    id: str
    kind: NodeKind
    name: str
    params: dict[str, ParamValue] = field(default_factory=dict)
    # Number of input ports for a variable-input operator (``sum``); ignored for fixed-arity kinds.
    n_inputs: int = 1
    # The id of the SBOL Component this node is grounded in, or ``None`` for an abstract node.
    component_id: str | None = None


@dataclass(frozen=True)
class Edge:
    """A directed wire from ``source`` to ``target``. ``port`` selects the target operator's input slot."""

    source: str
    target: str
    port: int = 0


@dataclass(frozen=True)
class GRN:
    """A gene-regulatory network: nodes and the edges wiring them."""

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()

    def species(self) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if not is_operator(node.kind))

    def operators(self) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if is_operator(node.kind))

    def node(self, node_id: str) -> Node:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(node_id)

    def inputs_of(self, operator_id: str) -> list[str]:
        """The source node ids feeding an operator, ordered by input port."""
        wired = [edge for edge in self.edges if edge.target == operator_id]
        return [edge.source for edge in sorted(wired, key=lambda e: e.port)]

    def outputs_of(self, operator_id: str) -> list[str]:
        """The species node ids an operator produces."""
        return [edge.target for edge in self.edges if edge.source == operator_id]

    def with_node(self, node: Node) -> GRN:
        return replace(self, nodes=self.nodes + (node,))

    def with_edge(self, edge: Edge) -> GRN:
        return replace(self, edges=self.edges + (edge,))

    def structural_hash(self) -> str:
        """A canonical key identifying the topology *and* parameters, invariant to node id and ordering.

        Two networks that differ only by how their nodes are named or listed share one key, so a search
        dedupes exactly-equivalent proposals (same wiring, same parameter values).
        """
        return self._canonical(include_params=True)

    def topology_hash(self) -> str:
        """A canonical key for the wiring alone, ignoring parameters, two circuits with the same parts
        and edges but different Hill values share it.

        This is the mode identity for portfolio diversity: distinct topologies are distinct *buildable*
        designs, while parameters are tuning within a topology. Canonical up to node naming (as
        :meth:`structural_hash`).
        """
        return self._canonical(include_params=False)

    def _canonical(self, *, include_params: bool) -> str:
        order = sorted(range(len(self.nodes)), key=lambda i: _node_key(self.nodes[i], include_params))
        index = {self.nodes[i].id: rank for rank, i in enumerate(order)}
        canon_nodes = [_node_key(self.nodes[i], include_params) for i in order]
        canon_edges = sorted(
            [index[e.source], index[e.target], e.port] for e in self.edges if e.source in index and e.target in index
        )
        return json.dumps({"nodes": canon_nodes, "edges": canon_edges}, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "name": n.name,
                    "params": n.params,
                    "inputCount": n.n_inputs,
                    **({"componentId": n.component_id} if n.component_id is not None else {}),
                }
                for n in self.nodes
            ],
            "edges": [{"source": e.source, "target": e.target, "port": e.port} for e in self.edges],
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> GRN:
        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("GRN.from_dict expects list-valued 'nodes' and 'edges'")
        nodes = tuple(
            Node(
                id=str(n["id"]),
                kind=_validated_kind(n["kind"]),
                name=str(n["name"]),
                params=dict(n.get("params", {})),
                n_inputs=int(n.get("inputCount", 1)),
                component_id=(str(n["componentId"]) if n.get("componentId") is not None else None),
            )
            for n in raw_nodes
        )
        edges = tuple(
            Edge(source=str(e["source"]), target=str(e["target"]), port=int(e.get("port", 0))) for e in raw_edges
        )
        return GRN(nodes=nodes, edges=edges)


def _node_key(node: Node, include_params: bool) -> list[object]:
    """A content signature for a node, used to canonicalize a hash. Parameters are included only for the
    exact (structural) hash, not the topology hash."""
    base: list[object] = [node.kind, node.name]
    if include_params:
        base.append(sorted((key, _clone(value)) for key, value in node.params.items()))
    return base
