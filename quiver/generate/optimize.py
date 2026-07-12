"""Topology-space search strategies, measured by the discovery curve: distinct high-scoring topologies
found per simulation budget. Random search and evolutionary search are the no-learning baselines a
learned diversity-seeking sampler must beat.

Each strategy returns a stream of ``(topology_hash, score)`` in evaluation order, the input to the
discovery-curve metric. Every candidate costs the same ``tune_steps + 1`` simulations (a small local
parameter search, then the score), so candidate index is proportional to simulation budget.
"""

from __future__ import annotations

import random

from quiver.generate.search import mutate
from quiver.generate.topology import random_sensor, structural_mutate
from quiver.grn import GRN
from quiver.objective import Behavior, ObjectiveFn, SimConfig, score_once

Scored = tuple[str, float]
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


def random_search_designs(
    behavior: Behavior,
    *,
    budget: int,
    tune_steps: int = 4,
    seed: int = 0,
    max_regulators: int = 4,
    objective: ObjectiveFn | None = None,
) -> list[Design]:
    """Random search keeping each candidate's tuned circuit, not only its hash and score."""
    rng = random.Random(seed)
    per = tune_steps + 1
    out: list[Design] = []
    for _ in range(max(1, budget // per)):
        grn = random_sensor(rng, max_regulators=max_regulators)
        out.append(tune_best(grn, behavior, rng, tune_steps, objective=objective))
    return out


def random_search(
    behavior: Behavior, *, budget: int, tune_steps: int = 4, seed: int = 0, max_regulators: int = 4
) -> list[Scored]:
    """Draw independent random topologies until the simulation budget is spent."""
    designs = random_search_designs(
        behavior, budget=budget, tune_steps=tune_steps, seed=seed, max_regulators=max_regulators
    )
    return [(grn.topology_hash(), score) for grn, score in designs]


def evolutionary_search_designs(
    behavior: Behavior,
    *,
    budget: int,
    tune_steps: int = 4,
    seed: int = 0,
    pop_size: int = 12,
    max_regulators: int = 4,
    objective: ObjectiveFn | None = None,
) -> list[Design]:
    """Evolutionary search keeping each evaluated candidate's tuned circuit, not only its hash and score.

    The population breeds from the pre-tuning topologies (structural moves act on the wiring and tuning
    only sets parameters) while the returned stream carries the tuned circuit for each evaluation."""
    rng = random.Random(seed)
    per = tune_steps + 1
    stream: list[Design] = []
    evals = 0

    def evaluate(grn: GRN) -> float:
        nonlocal evals
        tuned, score = tune_best(grn, behavior, rng, tune_steps, objective=objective)
        evals += per
        stream.append((tuned, score))
        return score

    population: list[tuple[GRN, float]] = []
    for _ in range(pop_size):
        if evals >= budget:
            return stream
        seed_grn = random_sensor(rng, max_regulators=max_regulators)
        population.append((seed_grn, evaluate(seed_grn)))

    while evals < budget:
        population.sort(key=lambda pair: pair[1], reverse=True)
        survivors = population[: max(1, pop_size // 2)]
        children: list[tuple[GRN, float]] = []
        for parent, _ in survivors:
            if evals >= budget:
                break
            child = structural_mutate(parent, rng)
            children.append((child, evaluate(child)))
        population = survivors + children
    return stream


def evolutionary_search(
    behavior: Behavior, *, budget: int, tune_steps: int = 4, seed: int = 0, pop_size: int = 12, max_regulators: int = 4
) -> list[Scored]:
    """Keep the top half of a population, breed offspring by structural mutation, repeat until the budget
    is spent. Beats random only if good topologies have good structural neighbors."""
    designs = evolutionary_search_designs(
        behavior, budget=budget, tune_steps=tune_steps, seed=seed, pop_size=pop_size, max_regulators=max_regulators
    )
    return [(grn.topology_hash(), score) for grn, score in designs]
