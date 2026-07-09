from __future__ import annotations

import pytest

pytest.importorskip("loica")

from quiver.generate import evolutionary_search, random_search  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402


@pytest.mark.sim
def test_searches_return_a_scored_stream() -> None:
    behavior = dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0))
    rnd = random_search(behavior, budget=15, tune_steps=2, seed=0)
    evo = evolutionary_search(behavior, budget=15, tune_steps=2, seed=0, pop_size=4)
    for stream in (rnd, evo):
        assert stream
        for topology, score in stream:
            assert isinstance(topology, str) and 0.0 <= score <= 1.0
