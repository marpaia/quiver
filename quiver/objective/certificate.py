"""Simulation-free structural certificates over the signed regulatory graph.

A certificate is a graph-theoretic predicate that is a proven *necessary* condition for a target dynamical
regime, read from the topology alone without integrating its kinetics:

- **Monotonicity** (Angeli and Sontag): a sign-consistent (balanced) interaction graph implies a monotone
  system, whose steady-state input/output response is unique and monotone. For a sensor this makes the
  dose-response well-posed, so the half-maximal crossing (EC50) is uniquely defined.
- **Thomas' rule** (Soule; Gouze; Snoussi): a positive feedback loop in the signed interaction graph is
  necessary for multistationarity (bistable, two-state memory); a negative loop is necessary for sustained
  oscillation.

Edge signs are read from operator kinetics. A single-input Hill or receiver operator with
``alpha = [basal, max]`` activates (``+1``) when ``max > basal`` and represses (``-1``) otherwise. A
two-input ``hill2`` operator contributes a per-input sign read from its four-entry response vector
``[a0, a1, a2, a3]``, whose entries are the expression at (both inputs off), (port-0 input on), (port-1
input on), and (both on) --- the ordering LOICA integrates, ``a0 + a1*r1 + a2*r2 + a3*r1*r2`` with ``r1``
the port-0 input. Each input's sign is ``+1``/``-1`` when the output is monotone in that input, ``0`` when
it is non-monotone (e.g. an XOR-like logic), and no edge when the input has no effect.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from quiver.grn import GRN, Node

# A signed edge of the species-level interaction graph: (source species, target species, sign in {-1,0,+1}).
SignedEdge = tuple[str, str, int]


def _pair_sign(basal: float, maximum: float) -> int | None:
    """Sign of a single-input Hill/receiver operator from its ``[basal, max]`` expression bounds."""
    if maximum > basal:
        return 1
    if maximum < basal:
        return -1
    return None


def _sign_from_deltas(deltas: list[float]) -> int | None:
    """Monotone sign of an input from its output changes across the other input's states.

    Returns ``+1`` if every change is non-negative (and some positive), ``-1`` if every change is
    non-positive (and some negative), ``None`` if the input has no effect, and ``0`` if the effect changes
    sign (non-monotone).
    """
    positive = any(d > 0.0 for d in deltas)
    negative = any(d < 0.0 for d in deltas)
    if positive and negative:
        return 0
    if positive:
        return 1
    if negative:
        return -1
    return None


def _operator_input_signs(node: Node) -> list[int | None]:
    """Per-input-port sign of an operator's effect on the species it produces."""
    alpha = node.params.get("alpha")
    if node.kind in ("receiver", "hill1"):
        if isinstance(alpha, list) and len(alpha) >= 2:
            pair = [float(x) for x in alpha[:2] if isinstance(x, (int, float))]
            if len(pair) == 2:
                return [_pair_sign(pair[0], pair[1])]
        return [None]
    if node.kind == "hill2":
        if isinstance(alpha, list) and len(alpha) >= 4:
            a = [float(x) for x in alpha[:4] if isinstance(x, (int, float))]
            if len(a) == 4:
                in0 = _sign_from_deltas([a[1] - a[0], a[3] - a[2]])
                in1 = _sign_from_deltas([a[2] - a[0], a[3] - a[1]])
                return [in0, in1]
        return [None, None]
    if node.kind == "sum":
        return [1] * max(1, node.n_inputs)
    return []


def signed_interaction_graph(grn: GRN) -> list[SignedEdge]:
    """Collapse operators to a signed, directed species-to-species interaction graph.

    Each operator contributes one signed edge from every input species to every species it produces. Inputs
    with no effect are dropped; non-monotone inputs carry sign ``0``.
    """
    edges: list[SignedEdge] = []
    for op in grn.operators():
        signs = _operator_input_signs(op)
        inputs = grn.inputs_of(op.id)
        outputs = grn.outputs_of(op.id)
        for port, src in enumerate(inputs):
            sign = signs[port] if port < len(signs) else None
            if sign is None:
                continue
            for dst in outputs:
                edges.append((src, dst, sign))
    return edges


def is_monotone(grn: GRN) -> bool:
    """Whether the interaction graph is sign-consistent (balanced), the Angeli-Sontag monotonicity
    certificate. A non-monotone operator edge (sign ``0``) or an incoherent reconvergence (two paths of
    opposite sign between the same species) breaks consistency and returns ``False``.
    """
    edges = signed_interaction_graph(grn)
    if any(sign == 0 for _, _, sign in edges):
        return False
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for src, dst, sign in edges:
        adjacency[src].append((dst, sign))
        adjacency[dst].append((src, sign))
    spin: dict[str, int] = {}
    for root in adjacency:
        if root in spin:
            continue
        spin[root] = 1
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor, sign in adjacency[node]:
                want = sign * spin[node]
                if neighbor not in spin:
                    spin[neighbor] = want
                    stack.append(neighbor)
                elif spin[neighbor] != want:
                    return False
    return True


def _simple_cycle_signs(edges: list[SignedEdge]) -> list[int]:
    """Sign product of every directed simple cycle. A cycle through a non-monotone (sign ``0``) edge has
    product ``0`` and is neither positive nor negative."""
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    nodes: set[str] = set()
    for src, dst, sign in edges:
        adjacency[src].append((dst, sign))
        nodes.add(src)
        nodes.add(dst)
    signs: list[int] = []
    for start in sorted(nodes):
        stack: list[tuple[str, list[str], int]] = [(start, [start], 1)]
        while stack:
            node, path, product = stack.pop()
            for neighbor, sign in adjacency[node]:
                if neighbor == start:
                    signs.append(product * sign)
                elif neighbor > start and neighbor not in path:
                    stack.append((neighbor, path + [neighbor], product * sign))
    return signs


def loop_signs(grn: GRN) -> tuple[bool, bool]:
    """``(has_positive_loop, has_negative_loop)`` over the signed interaction graph.

    A positive loop is necessary for multistationarity (bistable memory); a negative loop is necessary for
    sustained oscillation (Thomas' rule).
    """
    signs = _simple_cycle_signs(signed_interaction_graph(grn))
    return (any(s > 0 for s in signs), any(s < 0 for s in signs))


def certifies(grn: GRN, regime: str) -> bool:
    """Whether the topology carries the necessary-condition certificate for a target ``regime``.

    ``"sensor"`` requires interaction-graph monotonicity (a well-posed, unique monotone dose-response);
    ``"switch"`` requires a positive feedback loop (multistationarity capacity); ``"oscillator"`` requires
    a negative feedback loop (sustained-oscillation capacity).
    """
    if regime == "sensor":
        return is_monotone(grn)
    has_positive, has_negative = loop_signs(grn)
    if regime == "switch":
        return has_positive
    if regime == "oscillator":
        return has_negative
    raise ValueError(f"unknown regime: {regime!r}")


def monotone_gate(penalty: float = 0.25) -> Callable[[GRN], float]:
    """A multiplicative reward gate that down-weights topologies failing the monotonicity certificate.

    Certified topologies keep their full reward; the rest are scaled by ``penalty`` (in ``[0, 1)``), so a
    diversity-seeking policy trained under the gate steers its mass toward structurally well-posed designs
    without a hard feasibility wall that would erase the gradient.
    """

    def gate(grn: GRN) -> float:
        return 1.0 if is_monotone(grn) else penalty

    return gate
