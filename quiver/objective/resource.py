"""The metabolic-burden discount: a cheap analytic penalty on a circuit's total expression demand.

A finite shared expression budget (RNA polymerase, ribosomes) is drawn on by every gene, so a deeper,
higher-expression cascade loads the host harder. :func:`burden_fraction` turns a circuit's total demand into
a discount in ``[0, 1)``, the buildability term the default objective multiplies into a candidate's score
(:func:`quiver.objective.score.buildable_objective`) --- topology-dependent through :func:`circuit_demand`,
and requiring no extra simulation.
"""

from __future__ import annotations

from quiver.grn import GRN, ParamValue


def _alpha_values(params: dict[str, ParamValue]) -> list[float]:
    """The numeric entries of an operator's ``alpha`` expression vector."""
    alpha = params.get("alpha")
    if isinstance(alpha, list):
        return [float(x) for x in alpha if isinstance(x, (int, float))]
    return []


def circuit_demand(grn: GRN) -> float:
    """Total peak expression the circuit draws from the shared pool: summed over operators, each operator's
    maximum expression (the ``max`` of its ``alpha`` vector) plus any constitutive source ``rate``."""
    total = 0.0
    for op in grn.operators():
        values = _alpha_values(op.params)
        if values:
            total += max(values)
        rate = op.params.get("rate")
        if isinstance(rate, (int, float)):
            total += float(rate)
    return total


def burden_fraction(grn: GRN, capacity: float) -> float:
    """The fraction of expression dynamic range lost to resource sharing, ``phi = J / (1 + J)`` with total
    demand ``J = circuit_demand / capacity``. Zero when nothing competes, approaching one as demand swamps
    the pool; topology-dependent through :func:`circuit_demand`."""
    if capacity <= 0.0:
        return 0.0
    j = circuit_demand(grn) / capacity
    return j / (1.0 + j)
