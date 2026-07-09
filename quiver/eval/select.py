"""Diversity-select: turn a scored population into a portfolio of distinct, high-scoring designs.

The deliverable of a design run is a diverse batch, not a single argmax. This keeps one representative per
distinct topology, the highest-scoring parameterization found for it, ranked by score. Distinct
topologies are distinct buildable designs, so the result is diverse by construction; a max-min distance
spread within the retained set is a later refinement.
"""

from __future__ import annotations

from typing import Sequence


def select_portfolio(scored: Sequence[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """From ``(topology_hash, score)`` pairs, the top ``k`` distinct topologies by their best score."""
    best: dict[str, float] = {}
    for topology, score in scored:
        if topology not in best or score > best[topology]:
            best[topology] = score
    return sorted(best.items(), key=lambda item: item[1], reverse=True)[:k]
