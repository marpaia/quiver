from __future__ import annotations

from quiver.eval import select_portfolio


def test_select_keeps_best_per_topology_ranked_by_score() -> None:
    scored = [("A", 0.2), ("A", 0.9), ("B", 0.5), ("C", 0.7)]
    assert select_portfolio(scored, k=2) == [("A", 0.9), ("C", 0.7)]


def test_select_handles_empty_and_small_k() -> None:
    assert select_portfolio([], 3) == []
    assert select_portfolio([("A", 0.5)], 5) == [("A", 0.5)]
