"""The local parameter tuning primitive: given a topology, search its parameter neighborhood for the
best-scoring parameterization. This is the per-candidate tuning the learned sampler runs after proposing a
topology; the non-learned topology-search baselines built on it live in
:mod:`quiver.research.baselines.search_baselines`.
"""

from __future__ import annotations

import random

from quiver.generate.search import mutate
from quiver.grn import GRN
from quiver.objective import Behavior, ObjectiveFn, SimConfig, score_once

Design = tuple[GRN, float]


def tune_best(
    grn: GRN,
    behavior: Behavior,
    rng: random.Random,
    steps: int,
    config: SimConfig | None = None,
    *,
    objective: ObjectiveFn | None = None,
) -> Design:
    """A small local parameter search: the best-scoring parameterization found over ``steps`` mutations,
    returned with its score. ``config`` selects the simulation mode, the default dose-response window for
    sensors, or a long constant-growth window (`long_run`) for autonomous dynamics like oscillation.

    ``objective`` overrides how a candidate is scored: given, it is a circuit-to-scalar (e.g.
    :func:`quiver.objective.buildable_objective`, robustness discounted by burden) evaluated directly;
    absent, a single ``score_once`` shape-match at the candidate's parameters."""
    evaluate = objective if objective is not None else (lambda candidate: score_once(candidate, behavior, config))
    best = evaluate(grn)
    current = best_grn = grn
    for _ in range(steps):
        candidate = mutate(current, rng, jitter=0.4)
        score = evaluate(candidate)
        if score > best:
            current = best_grn = candidate
            best = score
    return best_grn, best


def tune_score(
    grn: GRN,
    behavior: Behavior,
    rng: random.Random,
    steps: int,
    config: SimConfig | None = None,
    *,
    objective: ObjectiveFn | None = None,
) -> float:
    """The best score over ``steps`` parameter mutations, :func:`tune_best` without the tuned circuit."""
    return tune_best(grn, behavior, rng, steps, config, objective=objective)[1]
