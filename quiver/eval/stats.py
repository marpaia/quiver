"""Paired-comparison statistics for method comparisons across a distribution of targets.

Pooling distinct-design counts as ``mean ± std`` over seeds × targets mixes two sources of spread: how
hard each target is (large, and irrelevant to which sampler wins) and seed-to-seed noise (the thing the
comparison is about). Because every method is run on the same (seed, target) cells, the comparison is
naturally paired: the within-cell difference cancels target difficulty, so the paired advantage and its
win rate are the defensible signal where the pooled standard deviation is not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb, sqrt
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class PairedSummary:
    """The advantage of method ``a`` over method ``b`` across aligned (seed, target) cells.

    ``mean_diff`` is the mean within-cell difference ``a - b``. ``win_rate`` is the fraction of cells where
    ``a`` is at least ``b``; ``strict_win_rate`` where ``a`` strictly exceeds ``b``. ``ci_low``/``ci_high``
    bound ``mean_diff`` with a bootstrap percentile interval. ``sign_p`` is the two-sided sign-test p-value
    that ``a`` and ``b`` are drawn from the same distribution (ties excluded).
    """

    n: int
    mean_diff: float
    win_rate: float
    strict_win_rate: float
    ci_low: float
    ci_high: float
    sign_p: float


def bootstrap_ci(
    values: Sequence[float], *, seed: int = 0, resamples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """A bootstrap percentile confidence interval for the mean of ``values`` at level ``1 - alpha``."""
    data = list(values)
    if not data:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(data)
    means = sorted(mean(data[rng.randrange(n)] for _ in range(n)) for _ in range(max(1, resamples)))
    lo = means[int((alpha / 2) * (len(means) - 1))]
    hi = means[int((1 - alpha / 2) * (len(means) - 1))]
    return (lo, hi)


@dataclass(frozen=True)
class RandomEffects:
    """A DerSimonian-Laird random-effects pooled estimate over per-target effects.

    ``mean`` is the pooled effect, weighting each target by the inverse of its total variance (within-target
    noise plus the estimated between-target heterogeneity ``tau2``); ``ci_low``/``ci_high`` bound it at the
    95% level. ``tau2`` is the between-target variance and ``i2`` the fraction of total variation due to
    heterogeneity rather than within-target noise (``0`` = targets homogeneous, near ``1`` = the effect
    varies strongly across targets). Pooling this way, rather than an unweighted bootstrap over targets,
    is the honest summary when targets differ in both effect size and precision.
    """

    k: int
    mean: float
    ci_low: float
    ci_high: float
    tau2: float
    i2: float


def random_effects(effects: Sequence[float], variances: Sequence[float], *, z: float = 1.96) -> RandomEffects:
    """Pool per-target ``effects`` with their within-target ``variances`` by the DerSimonian-Laird
    random-effects model. Non-positive variances are floored to a tiny positive value so a zero-variance
    target does not take infinite weight."""
    pairs = [(e, max(v, 1e-9)) for e, v in zip(effects, variances)]
    k = len(pairs)
    if k == 0:
        return RandomEffects(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if k == 1:
        effect, var = pairs[0]
        se = sqrt(var)
        return RandomEffects(1, effect, effect - z * se, effect + z * se, 0.0, 0.0)
    weights = [1.0 / var for _, var in pairs]
    total_w = sum(weights)
    fixed = sum(w * e for w, (e, _) in zip(weights, pairs)) / total_w
    q = sum(w * (e - fixed) ** 2 for w, (e, _) in zip(weights, pairs))
    df = k - 1
    c = total_w - sum(w * w for w in weights) / total_w
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    re_weights = [1.0 / (var + tau2) for _, var in pairs]
    total_rw = sum(re_weights)
    pooled = sum(w * e for w, (e, _) in zip(re_weights, pairs)) / total_rw
    se = sqrt(1.0 / total_rw)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return RandomEffects(k, pooled, pooled - z * se, pooled + z * se, tau2, i2)


def sign_test_p(a: Sequence[float], b: Sequence[float]) -> float:
    """The exact two-sided sign-test p-value for paired samples ``a`` and ``b`` (tied cells excluded)."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def paired_summary(
    a: Sequence[float], b: Sequence[float], *, seed: int = 0, resamples: int = 2000, alpha: float = 0.05
) -> PairedSummary:
    """Summarize the paired advantage of ``a`` over ``b`` across aligned cells.

    ``a`` and ``b`` must be aligned (the same (seed, target) cell at each index); a length mismatch is
    truncated to the shorter. Returns an all-zero summary for empty input.
    """
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return PairedSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    lo, hi = bootstrap_ci(diffs, seed=seed, resamples=resamples, alpha=alpha)
    return PairedSummary(
        n=len(diffs),
        mean_diff=mean(diffs),
        win_rate=sum(1 for d in diffs if d >= 0) / len(diffs),
        strict_win_rate=sum(1 for d in diffs if d > 0) / len(diffs),
        ci_low=lo,
        ci_high=hi,
        sign_p=sign_test_p(a, b),
    )
