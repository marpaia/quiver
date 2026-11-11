"""The conditioned sampler: one policy trained across several briefs, sampled for any of them."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("loica")

from quiver.learn.amortized import (  # noqa: E402
    SUMMARY_FEATURES,
    Brief,
    conditioned_policy,
    sample_conditioned,
    train_conditioned,
)
from quiver.objective import inducible_response  # noqa: E402


def _briefs() -> list[Brief]:
    up = inducible_response("GFP", direction="up")
    down = inducible_response("GFP", direction="down")
    return [
        Brief(encoding=(1.0, 0.0, 0.5), inducer="aTc", behavior=up),
        Brief(encoding=(0.0, 1.0, 0.25), inducer="aTc", behavior=down),
    ]


def test_policy_is_sized_for_the_conditioning_width() -> None:
    policy = conditioned_policy(7)
    assert list(policy.net)[0].in_features == SUMMARY_FEATURES + 7


def test_training_needs_at_least_one_brief() -> None:
    with pytest.raises(ValueError, match="at least one brief"):
        train_conditioned([])


@pytest.mark.sim
def test_one_policy_samples_every_brief_it_trained_on() -> None:
    briefs = _briefs()
    policy = train_conditioned(briefs, steps=3, batch=4, max_steps=5, seed=0)
    for brief in briefs:
        terminals = sample_conditioned(policy, brief, 4, max_steps=5, seed=1)
        assert len(terminals) == 4
        for grn in terminals:
            kinds = {n.kind for n in grn.nodes}
            assert "supplement" in kinds and "reporter" in kinds


@pytest.mark.sim
def test_a_brief_held_out_of_training_can_still_be_sampled() -> None:
    # The amortization claim: conditioning is a vector, so a request the policy never trained on is a
    # nearby point rather than an unknown label.
    trained, held_out = _briefs()
    policy = train_conditioned([trained], steps=3, batch=4, max_steps=5, seed=0)
    terminals = sample_conditioned(policy, held_out, 3, max_steps=5, seed=1)
    assert len(terminals) == 3
