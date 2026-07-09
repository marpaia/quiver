from __future__ import annotations

import pytest

# The whole module needs the LOICA simulation objective; skip cleanly without the `sim` extra.
pytest.importorskip("loica")

from quiver.generate import example_inducible  # noqa: E402
from quiver.objective import inducible_response, score_once, simulate, to_loica  # noqa: E402


@pytest.mark.sim
def test_example_simulates_and_produces_a_dose_response() -> None:
    result = simulate(example_inducible())
    assert result.doses
    # Without Flapjack, LOICA labels the measured signal by the reporter's name.
    assert "GFP" in result.signals


@pytest.mark.sim
def test_to_loica_compiles_a_design_to_a_genetic_network() -> None:
    import loica as lc

    grn = example_inducible()
    network = to_loica(grn)
    assert isinstance(network, lc.GeneticNetwork)
    # Every GRN species and operator crosses into the LOICA network.
    assert [r.name for r in network.reporters] == ["GFP"]
    assert len(network.regulators) == sum(1 for n in grn.nodes if n.kind == "regulator")
    assert len(network.operators) == len(grn.operators())


@pytest.mark.sim
def test_example_scores_within_unit_interval() -> None:
    score = score_once(example_inducible(), inducible_response("GFP"))
    assert 0.0 <= score <= 1.0
