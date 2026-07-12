"""Train a structured-env policy, deliver a portfolio, and summarize its breadth and buildability.

The GNN benchmark and the objective-ablation study both train a policy over the structured action space,
sample a portfolio of terminals, tune each against the target, and then ask two questions of the delivered
set: how *broad* it is (distinct high-scoring topologies) and how *buildable* it is (burden, cascade depth,
realizable-parameter fraction). This module holds that shared train-deliver-summarize path so the two
experiments configure the sampler and score portfolios identically; only the policy module and the training
objective differ between them.

Tuning and the pass yardstick are the bare ``score_once`` shape-match, held fixed across arms so breadth is
comparable to the rest of the experiment suite and to each other; the training objective is what varies.
Buildability is read off the delivered (tuned) designs: burden is analytic, cascade depth is the regulator
count, and realizability is the Cello-envelope check.
"""

from __future__ import annotations

import random
from statistics import mean
from types import ModuleType

from quiver.generate import is_realizable, tune_best
from quiver.grn import GRN
from quiver.objective import Behavior, ObjectiveFn, burden_fraction

# The shared-resource capacity the burden discount is read against, matching ``buildable_objective``'s default.
BURDEN_CAPACITY = 500.0

Design = tuple[GRN, float]


def regulator_count(grn: GRN) -> int:
    """Cascade depth: the number of transcription-factor regulators, the structural axis the search grows."""
    return sum(1 for node in grn.nodes if node.kind == "regulator")


def deliver(
    module: ModuleType,
    behavior: Behavior,
    *,
    objective: ObjectiveFn | None,
    beta: float,
    seed: int,
    n_samples: int,
    tune_steps: int,
    train_steps: int,
    train_batch: int,
    max_steps: int,
    sample_offset: int = 1000,
) -> list[Design]:
    """Train ``module``'s policy under ``objective`` and return its tuned, delivered designs.

    ``module`` is :mod:`quiver.learn.structured` or :mod:`quiver.learn.gnn`; both expose the same ``train``
    and ``sample_terminals`` signatures, so a run over either is a head-to-head architecture comparison. The
    policy trains under ``objective`` (the buildability-aware reward, or ``None`` for the shape-match), then
    each sampled terminal is tuned against the shape-match yardstick so the delivered set is scored the same
    way regardless of the training signal."""
    policy = module.train(
        behavior,
        steps=train_steps,
        batch=train_batch,
        beta=beta,
        max_steps=max_steps,
        seed=seed,
        objective=objective,
    )
    sample_seed = sample_offset + seed
    terminals = module.sample_terminals(policy, n_samples, max_steps=max_steps, seed=sample_seed)
    rng = random.Random(sample_seed)
    return [tune_best(grn, behavior, rng, tune_steps) for grn in terminals]


def delivered_portfolio(designs: list[Design], pass_threshold: float) -> list[Design]:
    """The delivered portfolio: the best-scoring passing design of each distinct topology."""
    best: dict[str, Design] = {}
    for grn, score in designs:
        if score < pass_threshold:
            continue
        key = grn.topology_hash()
        if key not in best or score > best[key][1]:
            best[key] = (grn, score)
    return list(best.values())


def summarize(designs: list[Design], *, pass_threshold: float) -> dict[str, float]:
    """Breadth and buildability of a delivered set: distinct high-scoring topologies, their quality, and the
    portfolio's mean burden, cascade depth, and realizable-parameter fraction."""
    portfolio = delivered_portfolio(designs, pass_threshold)
    scores = [score for _, score in portfolio]
    return {
        "distinct": len(portfolio),
        "peak_score": max(scores) if scores else 0.0,
        "mean_score": mean(scores) if scores else 0.0,
        "burden_mean": mean(burden_fraction(grn, BURDEN_CAPACITY) for grn, _ in portfolio) if portfolio else 0.0,
        "regulators_mean": mean(regulator_count(grn) for grn, _ in portfolio) if portfolio else 0.0,
        "realizable_frac": mean(1.0 if is_realizable(grn) else 0.0 for grn, _ in portfolio) if portfolio else 0.0,
    }
