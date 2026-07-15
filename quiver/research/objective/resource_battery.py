"""Held-out failure model B: shared-resource competition (burden), as a decorrelation battery.

The context battery in :mod:`quiver.research.objective.stress` perturbs each design's parameters and assay
window --- the same Hill-kinetic model, shocked. This is a *mechanistically different* held-out failure
model: a finite shared expression budget (RNA polymerase, ribosomes) that every gene draws on, so expressing
one gene depresses the achievable output of the others. Its steady-state signature, in the shared-resource
models of Gyorgy et al. and Qian et al., is a *compression of achievable dynamic range* --- the ON level is
held down by the total demand on the pool while the leaky OFF level is not, so fold change collapses under
load. Unlike the parameter shocks this coupling is *topology-dependent*: a circuit with more, higher-expression
operators loads the pool harder, so whether structurally distinct topologies still decorrelate under it is a
separate question from decorrelation under parameter shocks.

The burden *discount* the shipped objective uses lives in :mod:`quiver.objective.resource`; this module applies
that same demand as a dynamic-range-compressing shock, for the held-out decorrelation study.
"""

from __future__ import annotations

from dataclasses import replace

from quiver.grn import GRN, OPERATOR_KINDS, Node, ParamValue
from quiver.objective.behavior import Behavior
from quiver.objective.resource import _alpha_values, burden_fraction
from quiver.research.objective.stress import Context, context_battery, score_under_context


def _compress(value: ParamValue, keep: float, floor: float) -> ParamValue:
    if isinstance(value, list):
        return [_compress(v, keep, floor) for v in value]  # type: ignore[return-value]
    if isinstance(value, (int, float)):
        return floor + keep * (float(value) - floor)
    return value


def apply_resource_competition(grn: GRN, capacity: float) -> GRN:
    """A copy of ``grn`` under a shared expression budget ``capacity``: every operator's expression range is
    compressed toward its leaky floor (the ``min`` of its ``alpha`` vector) by the burden fraction, and a
    source's constitutive rate toward zero. The floor is preserved and the ceiling pulled down, so fold
    change collapses under load --- the steady-state signature of resource sharing, applied identically to
    every design but felt in proportion to each design's own topology-dependent demand.
    """
    keep = 1.0 - burden_fraction(grn, capacity)

    def burden(node: Node) -> Node:
        params = dict(node.params)
        values = _alpha_values(params)
        if values and "alpha" in params:
            params["alpha"] = _compress(params["alpha"], keep, min(values))
        rate = params.get("rate")
        if isinstance(rate, (int, float)):
            params["rate"] = keep * float(rate)
        return replace(node, params=params)

    nodes = tuple(burden(n) if n.kind in OPERATOR_KINDS else n for n in grn.nodes)
    return replace(grn, nodes=nodes)


# A resource-stress condition: a shared-budget capacity paired with a background context. The battery
# crosses capacities with heterogeneous backgrounds so a design's failures across conditions are not a single
# monotone capacity threshold (which would correlate across all designs by construction) but a varied
# signature, the way the parameter-shock battery is --- which is what makes the failure-correlation split
# between same- and distinct-topology pairs well defined under this model.
ResourceCondition = tuple[float, Context]


def resource_battery() -> tuple[ResourceCondition, ...]:
    """The held-out resource-competition battery: shared-budget capacities crossed with distinct background
    contexts. Each condition applies a finite expression budget (a topology-dependent burden) under one
    background shock, so the conditions are heterogeneous rather than an ordered capacity ladder."""
    names = {"host-weak", "host-strong", "affinity-low", "dose-high", "grow-weak"}
    backgrounds = (Context("nominal"),) + tuple(c for c in context_battery() if c.name in names)
    capacities = (150.0, 400.0)
    return tuple((capacity, background) for capacity in capacities for background in backgrounds)


def score_under_resource(grn: GRN, behavior: Behavior, condition: ResourceCondition) -> float:
    """The behavior detector on ``grn`` under a resource-stress condition: the shared-budget burden applied
    first (compressing dynamic range by topology-dependent demand), then the background context shock."""
    capacity, context = condition
    return score_under_context(apply_resource_competition(grn, capacity), behavior, context)
