from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("loica")

from quiver.learn.gflownet import PolicyNet, discovery_stream, sample_terminals, train  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402


@pytest.mark.sim
def test_gflownet_trains_and_samples_valid_terminals() -> None:
    behavior = dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0))
    policy = train(behavior, steps=3, batch=4, max_steps=5, seed=0)
    assert isinstance(policy, PolicyNet)
    terminals = sample_terminals(policy, 5, max_steps=5, seed=1)
    assert len(terminals) == 5
    for grn in terminals:
        kinds = {n.kind for n in grn.nodes}
        assert "supplement" in kinds and "reporter" in kinds


@pytest.mark.sim
def test_sample_terminals_is_seed_deterministic() -> None:
    # The seed controls the sample: identical seeds reproduce it (idempotent), distinct seeds redraw it.
    behavior = dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0))
    policy = train(behavior, steps=3, batch=4, max_steps=5, seed=0)
    first = [g.topology_hash() for g in sample_terminals(policy, 6, max_steps=5, seed=7)]
    same = [g.topology_hash() for g in sample_terminals(policy, 6, max_steps=5, seed=7)]
    other = [g.topology_hash() for g in sample_terminals(policy, 6, max_steps=5, seed=8)]
    assert first == same
    assert first != other


@pytest.mark.sim
def test_discovery_stream_is_scored_and_deterministic() -> None:
    behavior = dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0))
    kw = dict(seed=0, steps=3, batch=4, max_steps=5, n_samples=6, tune_steps=1)
    stream = discovery_stream(behavior, **kw)
    assert len(stream) == 6
    assert all(isinstance(h, str) and 0.0 <= s <= 1.0 for h, s in stream)
    assert discovery_stream(behavior, **kw) == stream  # reproducible given the seed
