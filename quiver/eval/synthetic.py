"""A synthetic de-risk harness: feed the metrics sampling distributions with a known answer, so a
measurement bug cannot masquerade as a scientific result.

Two analytic limits stand in for the policies being compared: a *proportional* sampler that draws a mode
with probability proportional to its score (the diversity-seeking ideal, whose empirical frequencies
match the score-proportional target), and a *concentrated* sampler that places almost all mass on the
single highest-scoring mode (the argmax limit). A test asserts the metrics recover the textbook contrast.
"""

from __future__ import annotations

import random
from typing import Mapping


def proportional_samples(scores: Mapping[str, float], n: int, seed: int) -> list[str]:
    """Draw ``n`` modes with probability proportional to score."""
    rng = random.Random(seed)
    modes = list(scores)
    weights = [scores[mode] for mode in modes]
    return rng.choices(modes, weights=weights, k=n)


def concentrated_samples(scores: Mapping[str, float], n: int, seed: int, *, mass: float = 0.92) -> list[str]:
    """Draw ``n`` modes placing ``mass`` of the probability on the single highest-scoring mode."""
    rng = random.Random(seed)
    modes = list(scores)
    top = max(scores, key=lambda mode: scores[mode])
    return [top if rng.random() < mass else rng.choice(modes) for _ in range(n)]
