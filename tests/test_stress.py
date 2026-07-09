"""Held-out stress checks: shared-context parameter scaling and the fixed context battery."""

from __future__ import annotations

import pytest

from quiver.generate import example_inducible
from quiver.objective import Context, apply_context, context_battery
from quiver.objective.behavior import inducible_response
from quiver.objective.stress import score_under_context


def test_apply_context_scales_expression_and_affinity_leaving_others() -> None:
    grn = example_inducible()
    shocked = apply_context(grn, Context("t", alpha_mult=2.0, k_mult=3.0))
    receiver = shocked.node("rec")
    # alpha (a vector) scaled by alpha_mult, K (a scalar) by k_mult, cooperativity n untouched.
    assert receiver.params["alpha"] == [0.0, 200.0]
    assert receiver.params["K"] == 3.0
    assert receiver.params["n"] == 2.0
    # Degradation and initial concentration are neither expression nor affinity, so they are left alone.
    assert shocked.node("tetr").params["degradation_rate"] == 1.0


def test_apply_context_baseline_is_identity_on_scaled_keys() -> None:
    grn = example_inducible()
    same = apply_context(grn, Context("baseline"))
    assert same.node("rec").params["alpha"] == [0.0, 100.0]
    assert same.node("rec").params["K"] == 1.0


def test_context_battery_is_a_fixed_named_grid() -> None:
    battery = context_battery()
    assert len(battery) == 12
    assert len({c.name for c in battery}) == len(battery)
    assert all(isinstance(c, Context) for c in battery)


@pytest.mark.sim
def test_score_under_context_runs_and_stays_in_unit_interval() -> None:
    pytest.importorskip("loica")
    grn = example_inducible()
    behavior = inducible_response("gfp")
    for ctx in context_battery():
        score = score_under_context(grn, behavior, ctx)
        assert 0.0 <= score <= 1.0
