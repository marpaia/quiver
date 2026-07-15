"""Render a GRN design as a Mermaid ``flowchart`` for documentation and inspection.

The diagram mirrors the bipartite model: the inducer is a circle, the reporter a stadium, transcription-factor
regulators plain boxes, and the LOICA operators slanted boxes labelled by the part filling them (a grounded
operator shows its SBOL Component name, e.g. ``P1_PhlF``, an abstract one its operator kind). Edges are
labelled by role --- a supplement *senses* into a receiver, an operator *expresses* a species, and a regulator
*represses* (dashed) or *activates* (solid) the promoter it feeds, read from the operator's kinetics.
"""

from __future__ import annotations

from quiver.grn import GRN, Node, ParamValue, is_operator


def to_mermaid(grn: GRN) -> str:
    """A Mermaid ``flowchart LR`` string for ``grn``."""
    reg_labels = _regulator_labels(grn)
    lines = ["flowchart LR"]
    for node in grn.nodes:
        lines.append(f"  {node.id}{_shape(node, reg_labels)}")
    for edge in grn.edges:
        source, target = grn.node(edge.source), grn.node(edge.target)
        label, dashed = _edge_style(source, target)
        arrow = f"-. {label} .->" if dashed else f"-- {label} -->"
        lines.append(f"  {edge.source} {arrow} {edge.target}")
    return "\n".join(lines)


def _regulator_labels(grn: GRN) -> dict[str, str]:
    """Short display labels (``r0``, ``r1`` ...) for the generically-named intermediate regulators, in node
    order, so the diagram reads cleanly while the operators carry the real part identity."""
    labels: dict[str, str] = {}
    for node in grn.nodes:
        if node.kind == "regulator":
            labels[node.id] = f"r{len(labels)}"
    return labels


def _shape(node: Node, reg_labels: dict[str, str]) -> str:
    if node.kind == "supplement":
        return f'(("{node.name}"))'
    if node.kind == "reporter":
        return f'(["{node.name}"])'
    if is_operator(node.kind):
        return f'[/"{node.name}"/]'
    return f'["{reg_labels.get(node.id, node.name)}"]'


def _edge_style(source: Node, target: Node) -> tuple[str, bool]:
    """The ``(label, dashed)`` for an edge, from the roles of its endpoints."""
    if is_operator(target.kind):
        if source.kind == "supplement":
            return "senses", False
        sign = _operator_sign(target)
        if sign < 0:
            return "represses", True
        if sign > 0:
            return "activates", False
        return "regulates", False
    return "expresses", False


def _operator_sign(node: Node) -> float:
    """+1 activation, -1 repression, 0 otherwise, from a single-input operator's ``alpha`` bounds."""
    alpha = node.params.get("alpha")
    if node.kind in ("receiver", "hill1") and isinstance(alpha, list) and len(alpha) >= 2:
        lo, hi = _as_float(alpha[0]), _as_float(alpha[-1])
        if lo is not None and hi is not None:
            return 1.0 if hi > lo else (-1.0 if hi < lo else 0.0)
    return 0.0


def _as_float(value: ParamValue) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
