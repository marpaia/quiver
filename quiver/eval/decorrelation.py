"""Does structural diversity decorrelate downstream failure?

The discovery metric counts distinct topology hashes clearing the pass bar. Whether that count measures a
*risk hedge* (designs that fail for different reasons, so a filter rejecting one need not reject them all)
is a separate question. These functions answer it from failure vectors: for each design, which of a
shared battery of held-out conditions it fails.

The mechanism is *decorrelation*: two parameter variants of one topology tend to fail the same conditions
(their failures are correlated), while two distinct topologies fail different ones (decorrelated). A
portfolio of decorrelated members keeps at least one member alive across more conditions than a
correlated one of the same size, the property a diverse portfolio is claimed to have.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import fmean
from typing import Sequence

# True at a position means the design *failed* that held-out condition.
FailureVector = tuple[bool, ...]


def failed_conditions(scores: Sequence[float], threshold: float) -> FailureVector:
    """Turn per-condition scores into a failure vector: ``True`` where the score is below ``threshold``."""
    return tuple(score < threshold for score in scores)


def _correlation(a: FailureVector, b: FailureVector) -> float | None:
    """Pearson correlation of two aligned binary vectors; ``None`` if either has no variance (all-pass or
    all-fail), where correlation is undefined."""
    n = len(a)
    if n == 0 or len(b) != n:
        return None
    xa = [1.0 if v else 0.0 for v in a]
    xb = [1.0 if v else 0.0 for v in b]
    ma, mb = sum(xa) / n, sum(xb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    va = sum((x - ma) ** 2 for x in xa)
    vb = sum((y - mb) ** 2 for y in xb)
    if va <= 0.0 or vb <= 0.0:
        return None
    return float(cov / (va * vb) ** 0.5)


@dataclass(frozen=True)
class FailureCorrelation:
    """Mean pairwise failure correlation split by whether the pair shares a topology.

    ``same`` averages over pairs that are parameter variants of one topology; ``distinct`` over pairs of
    different topologies. Decorrelation is the claim ``distinct < same``. Pairs with an all-pass or
    all-fail member are excluded (correlation undefined), so ``n_same``/``n_distinct`` are the counts that
    entered each mean.
    """

    same: float | None
    distinct: float | None
    n_same: int
    n_distinct: int
    same_diffs: tuple[float, ...]
    distinct_diffs: tuple[float, ...]


def failure_correlation(items: Sequence[tuple[str, FailureVector]]) -> FailureCorrelation:
    """Mean pairwise failure correlation over ``(topology_hash, failure_vector)`` items, split by whether
    each pair shares its topology hash."""
    same: list[float] = []
    distinct: list[float] = []
    for (key_a, vec_a), (key_b, vec_b) in combinations(items, 2):
        corr = _correlation(vec_a, vec_b)
        if corr is None:
            continue
        (same if key_a == key_b else distinct).append(corr)
    return FailureCorrelation(
        same=fmean(same) if same else None,
        distinct=fmean(distinct) if distinct else None,
        n_same=len(same),
        n_distinct=len(distinct),
        same_diffs=tuple(same),
        distinct_diffs=tuple(distinct),
    )


def portfolio_survival(vectors: Sequence[FailureVector]) -> float:
    """Fraction of the held-out conditions the portfolio survives, a condition is survived if at least one
    member passes it. This is the probability a downstream filter drawn from the battery spares ≥ 1 design.
    """
    if not vectors:
        return 0.0
    n_conditions = len(vectors[0])
    if n_conditions == 0:
        return 0.0
    survived = sum(1 for c in range(n_conditions) if any(not vec[c] for vec in vectors))
    return survived / n_conditions


def marginal_pass_rate(vectors: Sequence[FailureVector]) -> float:
    """The mean per-member, per-condition pass rate across a set of designs, the quantity a survival
    comparison must be matched on, so a survival gap reflects decorrelation rather than stronger members.
    """
    passes = 0
    total = 0
    for vec in vectors:
        passes += sum(1 for failed in vec if not failed)
        total += len(vec)
    return passes / total if total else 0.0
