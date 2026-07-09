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
