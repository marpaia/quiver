from __future__ import annotations

import math

import pytest

from quiver.objective import long_run


def test_long_run_is_a_sustained_growth_window() -> None:
    cfg = long_run(sustained_growth=0.4, hours=120.0, n_measurements=240)
    assert cfg.sustained_growth == 0.4
    assert cfg.n_measurements == 240
    assert cfg.interval == pytest.approx(0.5)  # 120 h / 240 measurements


@pytest.mark.sim
def test_sustained_growth_keeps_autonomous_dynamics_bounded() -> None:
    # Under exponential growth a zero-degradation reporter blows up; the constant-biomass chemostat regime
    # must keep an autonomous circuit's trajectory finite and bounded.
    import random

    from quiver.generate import repressilator
    from quiver.objective import simulate

    result = simulate(repressilator(random.Random(0)), long_run())
    values = [v for matrix in result.signals.values() for trace in matrix for v in trace]
    assert values
    assert all(math.isfinite(v) and 0.0 <= v < 1e9 for v in values)


@pytest.mark.sim
def test_repressilator_sustains_oscillation_under_long_run() -> None:
    import random

    from quiver.generate import repressilator
    from quiver.objective import simulate, sustained_oscillation

    detect = sustained_oscillation("GFP")
    config = long_run(sustained_growth=1.0)
    scores = [detect(simulate(repressilator(random.Random(s)), config)) for s in range(6)]
    assert max(scores) >= 0.5  # a regime-seeded repressilator sustains detectable oscillation
