"""Structural generation of sensor topologies: inducible cascades of varying depth, from an inducer
through a chain of promoter/regulator stages to a reporter.

This is the source of *structural* diversity a portfolio needs, distinct wirings, not just parameter
variants of one motif. Depth is the structural axis here (a 1-, 2-, or 3-stage cascade is a distinct
buildable design, a distinct ``topology_hash``); activation vs repression at a stage is parameter tuning
within a topology. Branching, feedback, and two-input logic stages are the next increments of variety.
"""

from __future__ import annotations

import random
from dataclasses import replace

from quiver.grn import GRN, Edge, Node, NodeKind, ParamValue

_ARITY = {"source": 0, "receiver": 1, "hill1": 1, "hill2": 2}


def sensor_cascade(depth: int, rng: random.Random) -> GRN:
    """A linear inducible cascade of ``depth`` promoter stages from an inducer to a reporter.

    Stage 0 is a receiver reading the inducer; later stages are Hill promoters, each randomly activating
    or repressing; stages are linked by regulators, and the last stage drives the reporter.
    """
    depth = max(1, depth)
    nodes: list[Node] = [Node("sup", "supplement", "aTc", {})]
    edges: list[Edge] = []
    reporter = Node("out", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0})
    prev = "sup"
    for stage in range(depth):
        op_id = f"op{stage}"
        is_tail = stage == depth - 1
        kind: NodeKind
        params: dict[str, ParamValue]
        if stage == 0:
            kind = "receiver"
            params = {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}
        else:
            kind = "hill1"
            activating = rng.random() < 0.5
            params = {"alpha": [0.0, 100.0] if activating else [100.0, 0.0], "K": 1.0, "n": 2.0}
        nodes.append(Node(op_id, kind, op_id, params))
        edges.append(Edge(prev, op_id))
        produced = "out" if is_tail else f"reg{stage}"
        if not is_tail:
            nodes.append(Node(produced, "regulator", produced, {"degradation_rate": 1.0, "init_concentration": 0.0}))
        edges.append(Edge(op_id, produced))
        prev = produced
    nodes.append(reporter)
    return GRN(nodes=tuple(nodes), edges=tuple(edges))


def sensor_topologies(n: int, *, max_depth: int = 4, seed: int = 0) -> list[GRN]:
    """``n`` sensor cascades of depth drawn uniformly from ``1..max_depth``, a structurally diverse
    population whose distinct depths are distinct buildable designs."""
    rng = random.Random(seed)
    return [sensor_cascade(rng.randint(1, max_depth), rng) for _ in range(n)]


def random_sensor(
    rng: random.Random, *, max_regulators: int = 4, logic_prob: float = 0.4, source_prob: float = 0.5
) -> GRN:
    """A feedforward sensor circuit with genuine topological variety.

    The inducer drives a receiver; optionally a constitutive source contributes a constant input; then a
    random *acyclic* web of single-input (Hill1) and two-input logic (Hill2) promoters and regulators
    leads to the reporter. The combinatorial choice of regulator count, operator kinds, and input wiring
    grows the topology space super-exponentially in regulator count: small enough to enumerate at a few
    regulators but non-enumerable at the circuit scale the simulator supports, and in every regime too
    costly to score exhaustively under a simulation budget, the ingredient a learned sampler needs. Feedback
    (cyclic) motifs are a further increment; this generator stays feedforward so every circuit converges.
    """
    nodes: list[Node] = [Node("sup", "supplement", "aTc", {})]
    edges: list[Edge] = []
    available: list[str] = []
    counters = {"op": 0, "reg": 0}

    def add_reg() -> str:
        rid = f"reg{counters['reg']}"
        counters["reg"] += 1
        nodes.append(Node(rid, "regulator", rid, {"degradation_rate": 1.0, "init_concentration": 0.0}))
        return rid

    def add_op(kind: NodeKind, inputs: list[str], output: str) -> None:
        oid = f"op{counters['op']}"
        counters["op"] += 1
        nodes.append(Node(oid, kind, oid, _op_params(kind, rng)))
        for port, inp in enumerate(inputs):
            edges.append(Edge(inp, oid, port))
        edges.append(Edge(oid, output))

    r0 = add_reg()
    add_op("receiver", ["sup"], r0)
    available.append(r0)

    if rng.random() < source_prob:
        constant = add_reg()
        add_op("source", [], constant)
        available.append(constant)

    for _ in range(rng.randint(0, max_regulators - 1)):
        reg = add_reg()
        if len(available) >= 2 and rng.random() < logic_prob:
            add_op("hill2", rng.sample(available, 2), reg)
        else:
            add_op("hill1", [rng.choice(available)], reg)
        available.append(reg)

    nodes.append(
        Node("out", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0})
    )
    if len(available) >= 2 and rng.random() < logic_prob:
        add_op("hill2", rng.sample(available, 2), "out")
    else:
        add_op("hill1", [rng.choice(available)], "out")

    return GRN(nodes=tuple(nodes), edges=tuple(edges))


def diverse_topologies(
    n: int, *, seed: int = 0, max_regulators: int = 4, logic_prob: float = 0.4, source_prob: float = 0.5
) -> list[GRN]:
    """``n`` feedforward sensor circuits drawn from :func:`random_sensor`, a structurally diverse
    population spanning branched, two-input-logic, and constitutive-source motifs, not just cascades."""
    rng = random.Random(seed)
    return [
        random_sensor(rng, max_regulators=max_regulators, logic_prob=logic_prob, source_prob=source_prob)
        for _ in range(n)
    ]


# The structural move vocabulary shared by the evolutionary search and the constructive GFlowNet env.
SENSOR_MOVE_NAMES: tuple[str, ...] = ("relay", "rewire")


def apply_sensor_move(grn: GRN, move: str, rng: random.Random) -> GRN:
    """Apply a named structural move (``"relay"`` grafts a new regulator into the path to the reporter;
    ``"rewire"`` repoints an output stage's input), guarded so the result is always a valid sensor."""
    child = {"relay": _add_relay, "rewire": _rewire_output}[move](grn, rng)
    return child if _is_valid_sensor(child) else grn


def structural_mutate(grn: GRN, rng: random.Random) -> GRN:
    """A random topology-level edit from the shared move vocabulary. Returns the parent unchanged if the
    drawn move would break validity, so the caller always receives a valid, simulable sensor circuit.
    This is the move set an evolutionary search walks and the constructive GFlowNet builds over."""
    return apply_sensor_move(grn, rng.choice(SENSOR_MOVE_NAMES), rng)


def _is_valid_sensor(grn: GRN) -> bool:
    ids = {node.id for node in grn.nodes}
    kinds = {node.kind for node in grn.nodes}
    if "supplement" not in kinds or "reporter" not in kinds:
        return False
    if not all(node.source in ids and node.target in ids for node in grn.edges):
        return False
    for reporter in (n.id for n in grn.nodes if n.kind == "reporter"):
        if not any(edge.target == reporter for edge in grn.edges):
            return False
    for op in grn.operators():
        if len(grn.inputs_of(op.id)) != _ARITY[op.kind] or not grn.outputs_of(op.id):
            return False
    return True


def _output_operators(grn: GRN) -> list[str]:
    reporters = {n.id for n in grn.nodes if n.kind == "reporter"}
    return [op.id for op in grn.operators() if set(grn.outputs_of(op.id)) & reporters]


def _rewire_output(grn: GRN, rng: random.Random) -> GRN:
    """Point one input of an output-stage operator at a different existing regulator."""
    out_ops = _output_operators(grn)
    regulators = [n.id for n in grn.nodes if n.kind == "regulator"]
    if not out_ops or len(regulators) < 2:
        return grn
    op = rng.choice(out_ops)
    in_edges = [i for i, e in enumerate(grn.edges) if e.target == op]
    if not in_edges:
        return grn
    idx = rng.choice(in_edges)
    old = grn.edges[idx]
    choices = [r for r in regulators if r != old.source]
    if not choices:
        return grn
    edges = list(grn.edges)
    edges[idx] = Edge(rng.choice(choices), op, old.port)
    return replace(grn, edges=tuple(edges))


def _add_relay(grn: GRN, rng: random.Random) -> GRN:
    """Insert a new regulator produced by a Hill1 from an existing regulator, and route an output-stage
    operator through it, a valid, acyclic deepening of the path to the reporter."""
    regulators = [n.id for n in grn.nodes if n.kind == "regulator"]
    out_ops = _output_operators(grn)
    if not regulators or not out_ops:
        return grn
    suffix = f"{len(grn.nodes)}_{rng.randrange(1_000_000)}"
    reg_id, op_id = f"reg_{suffix}", f"op_{suffix}"
    nodes = grn.nodes + (
        Node(reg_id, "regulator", reg_id, {"degradation_rate": 1.0, "init_concentration": 0.0}),
        Node(op_id, "hill1", op_id, _op_params("hill1", rng)),
    )
    edges = list(grn.edges) + [Edge(rng.choice(regulators), op_id, 0), Edge(op_id, reg_id)]
    op = rng.choice(out_ops)
    in_edges = [i for i, e in enumerate(edges) if e.target == op]
    if in_edges:
        idx = rng.choice(in_edges)
        edges[idx] = Edge(reg_id, op, edges[idx].port)
    return GRN(nodes=nodes, edges=tuple(edges))


def _op_params(kind: NodeKind, rng: random.Random) -> dict[str, ParamValue]:
    if kind == "source":
        return {"rate": 10.0}
    if kind == "hill1":
        activating = rng.random() < 0.5
        return {"alpha": [0.0, 100.0] if activating else [100.0, 0.0], "K": 1.0, "n": 2.0}
    if kind == "hill2":
        logic = rng.choice(
            [[0.0, 100.0, 100.0, 0.0], [100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0], [0.0, 100.0, 0.0, 100.0]]
        )
        return {"alpha": logic, "K": [1.0, 1.0], "n": [2.0, 2.0]}
    return {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}
