"""Shared-resource competition: demand accounting, burden fraction, range compression, and its effect on
the behavior score."""

from __future__ import annotations

import random

import pytest

from quiver.generate import example_inducible, random_sensor
from quiver.objective import burden_fraction, circuit_demand
from quiver.research.objective import apply_resource_competition, resource_battery


def test_circuit_demand_is_positive() -> None:
    assert circuit_demand(example_inducible()) > 0.0


def test_burden_fraction_decreases_with_capacity() -> None:
    grn = example_inducible()
    phis = [burden_fraction(grn, c) for c in (100.0, 300.0, 1000.0, 10000.0)]
    assert phis == sorted(phis, reverse=True)  # looser budget, less burden
    assert 0.0 <= phis[-1] < phis[0] < 1.0
    assert burden_fraction(grn, 0.0) == 0.0  # no pool defined => no burden


def test_burden_tracks_demand() -> None:
    a = random_sensor(random.Random(4), max_regulators=1)
    b = random_sensor(random.Random(5), max_regulators=12)
    cap = 300.0
    # the higher-demand circuit feels the greater burden at the same shared capacity.
    if circuit_demand(a) < circuit_demand(b):
        assert burden_fraction(a, cap) < burden_fraction(b, cap)
    else:
        assert burden_fraction(a, cap) >= burden_fraction(b, cap)


def test_apply_compresses_range_preserving_floor() -> None:
    grn = example_inducible()
    original = grn.node("rec").params["alpha"]
    tight = apply_resource_competition(grn, 100.0).node("rec").params["alpha"]
    loose = apply_resource_competition(grn, 1e9).node("rec").params["alpha"]
    assert isinstance(original, list) and isinstance(tight, list) and isinstance(loose, list)
    assert min(tight) == pytest.approx(min(original))  # leaky floor preserved
    assert max(tight) < max(original)  # ceiling pulled down under load
    assert loose == pytest.approx(original)  # ~identity when the pool is huge


def test_resource_battery_is_a_heterogeneous_grid() -> None:
    battery = resource_battery()
    assert len(battery) == 12  # 2 capacities x 6 backgrounds
    assert len({cap for cap, _ in battery}) == 2
    assert len({ctx.name for _, ctx in battery}) == 6


@pytest.mark.sim
def test_resource_burden_does_not_improve_the_score() -> None:
    pytest.importorskip("loica")
    from quiver.objective import score_once
    from quiver.objective.behavior import inducible_response
    from quiver.research.objective import Context, score_under_resource

    grn = example_inducible()
    behavior = inducible_response("gfp")
    loose = score_under_resource(grn, behavior, (1e9, Context("nominal")))
    tight = score_under_resource(grn, behavior, (150.0, Context("nominal")))
    assert 0.0 <= tight <= 1.0
    assert loose == pytest.approx(score_once(grn, behavior), abs=0.05)  # huge pool ~ no burden
    assert tight <= loose + 1e-6  # compressing dynamic range never helps
