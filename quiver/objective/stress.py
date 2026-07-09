"""Held-out stress checks: whether a design's behavior survives conditions it was never tuned against.

The training and tuning reward is the *nominal* detector score. A design's downstream worth, though, is
whether its behavior holds when the world differs from nominal: the part parameters differ from their
stated values (a build spans a range), and the cellular context differs (host strength, binding affinity,
growth, and the dose window an assay actually spans). These checks apply exactly those perturbations, so
they are orthogonal to what any sampler optimized and are the risk a diverse portfolio is meant to hedge.

Two families. *Parameter stress* perturbs each design's own numeric parameters independently, that is
:func:`quiver.objective.robustness`, reused directly, giving a per-design marginal survival
probability. A shared *context battery* instead applies one global shock identically to every design, so
whether two designs fail *the same* conditions is well defined, which is what makes decorrelation of
failure across a portfolio measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from quiver.grn import GRN, ParamValue
from quiver.objective.behavior import Behavior
from quiver.objective.score import score_once
from quiver.objective.simulate import SimConfig


@dataclass(frozen=True)
class Context:
    """One shared environmental condition applied identically to every design.

    ``alpha_mult`` scales every expression-rate parameter (``alpha`` and a source's ``rate``): a host
    strength / resource context. ``k_mult`` scales every half-saturation constant (``K``): a binding
    affinity context. ``config`` replaces the assay window and growth model (the dose range and metabolism
    the simulation spans). The baseline context (all multipliers ``1.0``, default config) reproduces the
    nominal score.
    """

    name: str
    alpha_mult: float = 1.0
    k_mult: float = 1.0
    config: SimConfig | None = None


def _scale(value: ParamValue, factor: float) -> ParamValue:
    if isinstance(value, list):
        return [_scale(entry, factor) for entry in value]  # type: ignore[return-value]
    if isinstance(value, (int, float)):
        return float(value) * factor
    return value


def apply_context(grn: GRN, ctx: Context) -> GRN:
    """A copy of ``grn`` with expression rates scaled by ``ctx.alpha_mult`` and half-saturation constants
    by ``ctx.k_mult``, the shared global shock, leaving cooperativity, degradation, and wiring intact."""

    def remap(params: dict[str, ParamValue]) -> dict[str, ParamValue]:
        out = dict(params)
        for key in ("alpha", "rate"):
            if key in out:
                out[key] = _scale(out[key], ctx.alpha_mult)
        if "K" in out:
            out["K"] = _scale(out["K"], ctx.k_mult)
        return out

    nodes = tuple(replace(node, params=remap(node.params)) for node in grn.nodes)
    return replace(grn, nodes=nodes)


def score_under_context(grn: GRN, behavior: Behavior, ctx: Context) -> float:
    """The behavior detector on ``grn`` simulated under the shared context ``ctx``."""
    return score_once(apply_context(grn, ctx), behavior, ctx.config)


# Dose windows shifted an order of magnitude either way from the default sweep, so a sensor tuned for an
# in-window EC50 can fall out of the window it is assayed over.
_DOSE_LOW = SimConfig(dose_min=1e-6, dose_max=1.0)
_DOSE_HIGH = SimConfig(dose_min=1e-2, dose_max=1e4)
# Growth models weaker and stronger than the default Gompertz (y0, ymax, um, lag), changing how much
# reporter accumulates over the assay.
_GROW_LOW = SimConfig(biomass=(0.05, 0.5, 0.5, 1.0))
_GROW_HIGH = SimConfig(biomass=(0.05, 2.0, 2.0, 1.0))


def context_battery() -> tuple[Context, ...]:
    """A fixed grid of shared context shocks. Each is applied identically to every design, so a design's
    failure vector over the battery is comparable across designs and topologies."""
    return (
        Context("host-weak", alpha_mult=0.3),
        Context("host-strong", alpha_mult=3.0),
        Context("affinity-low", k_mult=3.0),
        Context("affinity-high", k_mult=0.3),
        Context("weak+tight", alpha_mult=0.3, k_mult=0.3),
        Context("strong+loose", alpha_mult=3.0, k_mult=3.0),
        Context("dose-low", config=_DOSE_LOW),
        Context("dose-high", config=_DOSE_HIGH),
        Context("grow-weak", config=_GROW_LOW),
        Context("grow-strong", config=_GROW_HIGH),
        Context("weak+dose-low", alpha_mult=0.3, config=_DOSE_LOW),
        Context("strong+dose-high", alpha_mult=3.0, config=_DOSE_HIGH),
    )
