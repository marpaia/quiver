"""The objective a design search optimizes: how robustly a circuit realizes a target behavior.

A single simulation scores one parameterization. Because a real build spans a range of part parameters, a
circuit's worth is its behavior's *robustness* over a parameter ensemble, the mean detector score across
perturbed parameterizations. That ensemble average is the dense, quantitative signal the search follows.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, cast

from quiver.grn import GRN, ParamValue
from quiver.objective.behavior import Behavior
from quiver.objective.simulate import SimConfig, simulate

# The objective a generator is scored against: a circuit to a scalar in [0, 1].
ObjectiveFn = Callable[[GRN], float]


def score_once(grn: GRN, behavior: Behavior, config: SimConfig | None = None) -> float:
    """The behavior detector on a single simulation at the circuit's stated parameters."""
    return behavior(simulate(grn, config))


def robustness(
    grn: GRN,
    behavior: Behavior,
    *,
    config: SimConfig | None = None,
    ensemble: int = 16,
    jitter: float = 0.3,
    seed: int = 0,
) -> float:
    """Mean detector score over an ensemble of log-normally perturbed parameterizations of ``grn``.

    ``jitter`` is the standard deviation of the log-normal multiplier applied to every numeric parameter;
    ``ensemble`` is the number of perturbed simulations averaged. The stated parameters are included as
    the first ensemble member, so a fragile design that works only at its nominal point scores low.
    """
    rng = random.Random(seed)
    total = score_once(grn, behavior, config)
    for _ in range(max(0, ensemble - 1)):
        total += score_once(_perturb(grn, rng, jitter), behavior, config)
    return total / max(1, ensemble)


def robustness_objective(
    behavior: Behavior,
    *,
    config: SimConfig | None = None,
    ensemble: int = 16,
    jitter: float = 0.3,
    seed: int = 0,
) -> ObjectiveFn:
    """Bind a behavior detector into an :data:`ObjectiveFn` a generator can be scored against."""

    def objective(grn: GRN) -> float:
        return robustness(grn, behavior, config=config, ensemble=ensemble, jitter=jitter, seed=seed)

    return objective


def buildable_objective(
    behavior: Behavior,
    *,
    config: SimConfig | None = None,
    ensemble: int = 8,
    jitter: float = 0.3,
    seed: int = 0,
    burden_capacity: float | None = 500.0,
) -> ObjectiveFn:
    """The default *buildability-aware* design objective: how robustly a circuit realizes the target
    behavior, discounted by the metabolic burden its topology imposes. This is the dense signal the search
    follows, so a design is rewarded for a *buildable* realization of the target rather than a single lucky
    parameterization.

    Terms, multiplied into a score in ``[0, 1]``:

    - **Robustness**, the mean detector score over ``ensemble`` log-normally perturbed parameterizations
      (:func:`robustness`); the primary term.
    - **Burden**, a discount of ``1 - burden_fraction(grn, burden_capacity)``
      (:func:`quiver.objective.resource.burden_fraction`): a cheap analytic penalty that grows with the
      circuit's total expression demand, so deeper, higher-expression cascades are penalized *without any
      extra simulation*. ``burden_capacity=None`` disables it.
    """

    def objective(grn: GRN) -> float:
        score = robustness(grn, behavior, config=config, ensemble=ensemble, jitter=jitter, seed=seed)
        if burden_capacity is not None:
            from quiver.objective.resource import burden_fraction

            score *= 1.0 - burden_fraction(grn, burden_capacity)
        return score

    return objective


def _perturb(grn: GRN, rng: random.Random, jitter: float) -> GRN:
    """A copy of ``grn`` with every numeric parameter scaled by an independent log-normal multiplier."""
    nodes = tuple(replace(node, params=_perturb_params(node.params, rng, jitter)) for node in grn.nodes)
    return replace(grn, nodes=nodes)


def _perturb_params(params: dict[str, ParamValue], rng: random.Random, jitter: float) -> dict[str, ParamValue]:
    return {key: _perturb_value(value, rng, jitter) for key, value in params.items()}


def _perturb_value(value: ParamValue, rng: random.Random, jitter: float) -> ParamValue:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value) * rng.lognormvariate(0.0, jitter)
    return cast(ParamValue, [_perturb_value(entry, rng, jitter) for entry in value])
