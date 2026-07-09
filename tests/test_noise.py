"""Intrinsic-noise failure model: replicate battery and reproducible stochastic scoring."""

from __future__ import annotations

import pytest

from quiver.generate import example_inducible
from quiver.objective import noise_battery


def test_noise_battery_is_distinct_replicate_seeds() -> None:
    battery = noise_battery(8)
    assert len(battery) == 8
    assert len(set(battery)) == 8


@pytest.mark.sim
def test_score_under_noise_is_reproducible_and_in_unit_interval() -> None:
    pytest.importorskip("loica")
    from quiver.objective.behavior import inducible_response
    from quiver.objective.noise import score_under_noise

    grn = example_inducible()
    behavior = inducible_response("gfp")
    a = score_under_noise(grn, behavior, 0)
    b = score_under_noise(grn, behavior, 0)
    c = score_under_noise(grn, behavior, 1)
    assert a == b  # same seed => identical realization
    assert 0.0 <= a <= 1.0 and 0.0 <= c <= 1.0
