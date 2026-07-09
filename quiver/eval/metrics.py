"""Evaluation metrics for a design run: reliability, portfolio breadth, and sampling quality.

Diversity, not peak score, is what decides the method here, so the headline quantities are the breadth of
the portfolio a generator surfaces (distinct valid modes discovered vs. budget) and how its sampling
tracks the score. A "mode" is one distinct valid design, identified by its structural hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log
from typing import Sequence


@dataclass(frozen=True)
class QualityReport:
    """Quality of the sampled valid designs, the axis a graded objective shapes and mode counts miss.

    ``mean_score`` is the average objective value over passers. ``score_tv`` is the total-variation
    distance between how often each found mode is sampled and the reward-proportional target (∝ its
    score\\ :sup:`β`, the exponent the sampler is trained toward) over the found modes: 0 is perfectly
    proportional sampling. ``top_mode_share`` is the fraction of
    passers on the single highest-scoring mode, which an argmax sampler drives toward 1. ``mode_entropy``
    is the normalized Shannon entropy of the sampling distribution over found modes: the portfolio's
    uniformity among what it found (1.0 even, near 0 one dominant mode). ``score_tv``/``top_mode_share``
    are ``None`` when there are no passers or every score is zero; ``mode_entropy`` is ``None`` below two
    found modes.
    """

    n_passers: int
    modes_hit: int
    mean_score: float
    score_tv: float | None
    top_mode_share: float | None
    mode_entropy: float | None


@dataclass(frozen=True)
class PortfolioReport:
    """Aggregate outcome of scoring a set of proposals for one target."""

    n: int
    passers: int
    distinct_passers: int
    total_modes: int | None = None
    discovery: tuple[int, ...] | None = None
    quality: QualityReport | None = None

    def pass_rate(self) -> float:
        return self.passers / self.n if self.n else 0.0

    def diversity(self) -> float:
        """Distinct passing designs per passer, near 0 is mode collapse, near 1 is varied output."""
        return self.distinct_passers / self.passers if self.passers else 0.0

    def coverage(self) -> float:
        """Fraction of the known valid modes covered, when ``total_modes`` is known (else 0)."""
        if not self.total_modes:
            return 0.0
        return min(self.distinct_passers, self.total_modes) / self.total_modes

    def modes_at(self, fraction: float) -> int | None:
        """Distinct valid modes discovered within the first ``fraction`` of the budget (a curve point)."""
        if not self.discovery:
            return None
        idx = max(0, min(len(self.discovery) - 1, round(fraction * len(self.discovery)) - 1))
        return self.discovery[idx]


def discovery_curve(hashes: Sequence[str | None], *, passed: Sequence[bool]) -> tuple[int, ...]:
    """Cumulative distinct valid modes discovered as a function of the number of proposals drawn, in order.

    When exhaustively scoring every topology is infeasible under a simulation budget, the *rate* of
    unique-mode discovery separates a portfolio-building generator from a concentrating one, where a static
    coverage fraction saturates.
    """
    seen: set[str] = set()
    curve: list[int] = []
    for structural_hash, ok in zip(hashes, passed):
        if ok and structural_hash is not None:
            seen.add(structural_hash)
        curve.append(len(seen))
    return tuple(curve)


def evaluate_quality(
    hashes: Sequence[str | None], scores: Sequence[float], *, passed: Sequence[bool], beta: float = 1.0
) -> QualityReport:
    """Summarize the score distribution of the passing (valid) designs. The score is deterministic per
    structure, so a mode's score is well-defined. ``beta`` sets the exponent of the reward-proportional
    target the ``score_tv`` diagnostic measures against (q_k ∝ score_k^β), matching the temper the sampler
    is trained toward."""
    pairs = [(h, s) for h, s, ok in zip(hashes, scores, passed) if ok and h is not None]
    if not pairs:
        return QualityReport(0, 0, 0.0, None, None, None)
    n = len(pairs)
    mean_score = sum(s for _, s in pairs) / n
    counts: dict[str, int] = {}
    mode_score: dict[str, float] = {}
    for structural_hash, score in pairs:
        counts[structural_hash] = counts.get(structural_hash, 0) + 1
        mode_score[structural_hash] = score
    entropy = _mode_entropy(counts)
    weights = {h: mode_score[h] ** beta for h in mode_score}
    total = sum(weights.values())
    if total <= 0.0:
        return QualityReport(n, len(counts), mean_score, None, None, entropy)
    score_tv = 0.5 * sum(abs(counts[h] / n - weights[h] / total) for h in counts)
    top = max(mode_score, key=lambda h: mode_score[h])
    return QualityReport(n, len(counts), mean_score, score_tv, counts[top] / n, entropy)


def _mode_entropy(counts: dict[str, int]) -> float | None:
    """Shannon entropy of the mode-sampling distribution, normalized to ``[0, 1]`` by ``log(#modes)``."""
    if len(counts) < 2:
        return None
    total = sum(counts.values())
    raw = -sum((c / total) * log(c / total) for c in counts.values())
    return raw / log(len(counts))


def evaluate_portfolio(
    scores: Sequence[float],
    hashes: Sequence[str | None],
    *,
    passed: Sequence[bool],
    total_modes: int | None = None,
    beta: float = 1.0,
) -> PortfolioReport:
    """Summarize a scored set of proposals: how many passed, how many distinct modes, and the curves.
    ``beta`` is passed to the sampling-quality diagnostic (the reward-proportional target's exponent)."""
    passing = [h for ok, h in zip(passed, hashes) if ok and h is not None]
    return PortfolioReport(
        n=len(scores),
        passers=len(passing),
        distinct_passers=len(set(passing)),
        total_modes=total_modes,
        discovery=discovery_curve(hashes, passed=passed),
        quality=evaluate_quality(hashes, scores, passed=passed, beta=beta),
    )


def sampling_fidelity(
    hashes: Sequence[str], scores: Sequence[float], *, threshold: float, beta: float
) -> tuple[float | None, float | None, float | None]:
    """Diagnostics on a sampler's distribution over topologies, estimated from a policy sample.

    Over the samples that pass ``threshold``, counts how often each topology is emitted (its empirical
    frequency) and its mean score across sampled instances --- the effective reward when an environment
    kernel fills the parameters the policy does not control. Returns normalized Shannon entropy of the
    topology frequencies, the top-mode mass, and the total-variation distance to the reward-proportional
    target :math:`q_k \\propto \\bar{r}_k^{\\beta}`. All three are ``None`` when fewer than two topologies
    pass (the distribution is degenerate); the TV is ``None`` when every mean reward is zero.
    """
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    for topology, score in zip(hashes, scores):
        if score < threshold:
            continue
        counts[topology] = counts.get(topology, 0) + 1
        totals[topology] = totals.get(topology, 0.0) + score
    n = sum(counts.values())
    if n == 0 or len(counts) < 2:
        return (None, None, None)
    entropy = -sum((c / n) * log(c / n) for c in counts.values()) / log(len(counts))
    top_mode = max(counts.values()) / n
    weights = {h: (totals[h] / counts[h]) ** beta for h in counts}
    wtotal = sum(weights.values())
    tv = 0.5 * sum(abs(counts[h] / n - weights[h] / wtotal) for h in counts) if wtotal > 0.0 else None
    return (entropy, top_mode, tv)


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimator (Chen et al.): the chance a size-``k`` draw contains a passer."""
    if k <= 0 or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
