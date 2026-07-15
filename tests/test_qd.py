"""Quality-diversity and Bayesian-optimization baselines: budget parity, validity, determinism."""

from __future__ import annotations

from typing import Callable

import pytest

pytest.importorskip("loica")

from quiver.objective import TargetResponse, dose_response_match
from quiver.research.baselines import gp_ucb_search, map_elites, novelty_search

Search = Callable[..., list[tuple[str, float]]]
TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
BUDGET = 30
TUNE = 2


@pytest.mark.sim
@pytest.mark.parametrize("search", [map_elites, novelty_search, gp_ucb_search])
def test_search_respects_budget_and_returns_valid_stream(search: Search) -> None:
    behavior = dose_response_match(TARGET)
    stream = search(behavior, budget=BUDGET, tune_steps=TUNE, seed=0)
    # Each candidate costs TUNE + 1 simulations, so the stream length matches the budget.
    assert len(stream) == BUDGET // (TUNE + 1)
    assert all(isinstance(h, str) and 0.0 <= s <= 1.0 for h, s in stream)


@pytest.mark.sim
@pytest.mark.parametrize("search", [map_elites, novelty_search, gp_ucb_search])
def test_search_is_deterministic_per_seed(search: Search) -> None:
    behavior = dose_response_match(TARGET)
    assert search(behavior, budget=BUDGET, tune_steps=TUNE, seed=1) == search(
        behavior, budget=BUDGET, tune_steps=TUNE, seed=1
    )
