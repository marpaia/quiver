"""A structure-controlling constructive MDP: the policy selects the choices the environment otherwise makes.

The env of :mod:`quiver.gfn.env` leaves the topology-determining choices of each structural move (which
existing regulator a ``relay`` grafts from, which regulator a ``rewire`` points to) to an environment kernel
the policy does not control, so its reward-proportionality guarantee holds over action *sequences* but only
up to that kernel over *topologies*. Here the policy chooses those regulators itself, by index into the
circuit's regulators in construction order, and the move is applied deterministically. With no environment
randomness, a topology is a deterministic function of the action sequence, so the sequence-to-topology gap
closes and the policy can shape the topology distribution directly (concentrating it as the reward exponent
grows, not only spreading it).

The other choices the original kernel made (activation vs. repression, node naming) are parameter and
naming level, invisible to :meth:`~quiver.grn.GRN.topology_hash`. A new stage's activation is drawn
exactly as the env-random generator draws it (behavior depends on it, and downstream tuning cannot flip it),
so the topology remains a deterministic function of the action sequence while parameters stay stochastic and
tuned, the same stance the main method takes toward parameters.
"""

from __future__ import annotations

import random
from dataclasses import replace

from quiver.generate.topology import _is_valid_sensor, _op_params, _output_operators
from quiver.grn import GRN, Edge, Node

# The largest number of regulators the indexed action head addresses; relay is masked out at this cap, which
# matches the ``max_regulators = 4`` of the random-sensor generator the baselines use.
MAX_REG = 4

# stop, plus relay-from-regulator-i and rewire-to-regulator-i for each addressable regulator index.
STRUCTURED_ACTIONS: tuple[str, ...] = (
    ("stop",) + tuple(f"relay:{i}" for i in range(MAX_REG)) + tuple(f"rewire:{i}" for i in range(MAX_REG))
)


def regulators_in_order(grn: GRN) -> list[str]:
    """The regulator ids in construction order (their order in the node tuple) --- the canonical indexing
    the action head addresses."""
    return [node.id for node in grn.nodes if node.kind == "regulator"]


def _canonical_output_input(grn: GRN) -> tuple[str, int] | None:
    """The output-stage operator and the index (into ``grn.edges``) of its first input edge --- the
    canonical site a ``relay`` reroutes through and a ``rewire`` repoints, chosen deterministically."""
    out_ops = sorted(_output_operators(grn))
    if not out_ops:
        return None
    op = out_ops[0]
    in_edges = sorted(i for i, e in enumerate(grn.edges) if e.target == op)
    if not in_edges:
        return None
    return op, in_edges[0]


def available_structured_actions(grn: GRN) -> list[str]:
    """The indexed moves valid from ``grn``: ``stop`` always; ``relay:i`` for each existing regulator while
    under the regulator cap; ``rewire:i`` for each regulator that would actually change the canonical output
    input (so every non-stop move strictly edits the topology)."""
    regs = regulators_in_order(grn)
    actions = ["stop"]
    if len(regs) < MAX_REG:
        actions.extend(f"relay:{i}" for i in range(len(regs)))
    site = _canonical_output_input(grn)
    if len(regs) >= 2 and site is not None:
        current = grn.edges[site[1]].source
        actions.extend(f"rewire:{i}" for i, r in enumerate(regs) if r != current)
    return actions


def apply_structured_action(grn: GRN, action: str, rng: random.Random) -> GRN:
    """Apply an indexed action. The topology it produces is a deterministic function of ``action``; ``rng``
    sets only the new stage's activation parameter (topology-invisible). ``stop`` returns ``grn`` unchanged
    (handled by the caller); an out-of-range or no-op index returns ``grn`` unchanged as well."""
    if action == "stop":
        return grn
    kind, raw = action.split(":")
    index = int(raw)
    regs = regulators_in_order(grn)
    if index >= len(regs):
        return grn
    if kind == "relay":
        return _structured_relay(grn, regs[index], rng)
    return _structured_rewire(grn, regs[index])


def _structured_relay(grn: GRN, source: str, rng: random.Random) -> GRN:
    """Graft a new regulator, produced by a Hill1 from ``source`` (activating or repressing, drawn as the
    env-random generator draws it), and route the canonical output-stage input through it."""
    site = _canonical_output_input(grn)
    if site is None:
        return grn
    op, edge_index = site
    suffix = len(grn.nodes)
    reg_id, op_id = f"reg_s{suffix}", f"op_s{suffix}"
    nodes = grn.nodes + (
        Node(reg_id, "regulator", reg_id, {"degradation_rate": 1.0, "init_concentration": 0.0}),
        Node(op_id, "hill1", op_id, _op_params("hill1", rng)),
    )
    edges = list(grn.edges) + [Edge(source, op_id, 0), Edge(op_id, reg_id)]
    edges[edge_index] = Edge(reg_id, op, grn.edges[edge_index].port)
    child = GRN(nodes=nodes, edges=tuple(edges))
    return child if _is_valid_sensor(child) else grn


def _structured_rewire(grn: GRN, target: str) -> GRN:
    """Repoint the canonical output-stage input at regulator ``target`` --- the deterministic form of the
    env's ``rewire``."""
    site = _canonical_output_input(grn)
    if site is None:
        return grn
    op, edge_index = site
    old = grn.edges[edge_index]
    if old.source == target:
        return grn
    edges = list(grn.edges)
    edges[edge_index] = Edge(target, op, old.port)
    child = replace(grn, edges=tuple(edges))
    return child if _is_valid_sensor(child) else grn
