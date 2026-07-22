from __future__ import annotations

from pathlib import Path

import pytest

from quiver import Designer, Portfolio, design
from quiver.generate import example_inducible
from quiver.grn import is_operator
from quiver.objective import TargetResponse, dose_response_match, sensor_target_grid


def _behavior() -> object:
    return dose_response_match(TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any"))


def test_portfolio_exposes_ranked_designs() -> None:
    grn = example_inducible()
    portfolio = Portfolio(((grn, 0.9), (grn, 0.5)))
    assert len(portfolio) == 2
    assert portfolio.best == (grn, 0.9)
    assert portfolio.scores == [0.9, 0.5]
    assert portfolio.circuits[0] is grn
    assert [score for _, score in portfolio] == [0.9, 0.5]


def test_design_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        design(_behavior(), policy="bogus")


@pytest.mark.sim
def test_design_returns_distinct_ranked_portfolio() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("loica")
    portfolio = design(_behavior(), policy="mlp", budget=40, tune_steps=1, threshold=0.0, seed=0)
    hashes = [grn.topology_hash() for grn, _ in portfolio]
    assert hashes, "expected at least one design"
    assert len(hashes) == len(set(hashes)), "portfolio must hold distinct topologies"
    assert portfolio.scores == sorted(portfolio.scores, reverse=True), "designs must be ranked best-first"


@pytest.mark.sim
def test_portfolio_networks_compile_to_loica() -> None:
    pytest.importorskip("torch")
    import loica as lc

    portfolio = design(_behavior(), policy="mlp", budget=40, tune_steps=1, threshold=0.0, seed=0)
    networks = portfolio.networks()
    assert len(networks) == len(portfolio)
    assert all(isinstance(network, lc.GeneticNetwork) for network in networks)


@pytest.mark.sim
def test_designer_trains_once_and_designs_many_targets() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("loica")
    grid = sensor_target_grid()
    designer = Designer.train(grid[1:], steps=6, batch=6, seed=0)

    # The same trained policy serves a trained target and a held-out one, with no retraining.
    trained = designer.design(grid[1], budget=24, tune_steps=1, threshold=0.0, seed=0)
    held_out = designer.design(grid[0], budget=24, tune_steps=1, threshold=0.0, seed=0)
    for portfolio in (trained, held_out):
        assert isinstance(portfolio, Portfolio)
        hashes = [grn.topology_hash() for grn, _ in portfolio]
        assert len(hashes) == len(set(hashes)), "portfolio must hold distinct topologies"


@pytest.mark.sim
def test_grounded_designer_designs_from_real_components(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("loica")
    from quiver import load_cello

    grid = sensor_target_grid()
    library = load_cello()
    designer = Designer.train(grid[1:], steps=6, batch=6, seed=0, library=library)

    # A held-out target is designed as a real bill-of-Components with no retraining.
    portfolio = designer.design(grid[0], budget=24, threshold=0.0, seed=0)
    assert len(portfolio) >= 1
    for grn, _ in portfolio:
        operators = [n for n in grn.nodes if is_operator(n.kind)]
        assert all(n.component_id is not None for n in operators), "delivered operators are real Components"

    # The grounding survives a save/load round trip: the reloaded designer is grounded in the same library.
    designer.save_pretrained(tmp_path)
    reloaded = Designer.from_pretrained(tmp_path)
    assert reloaded.library is not None and reloaded.library.name == "cello"


@pytest.mark.sim
def test_designer_save_pretrained_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("loica")
    designer = Designer.train(sensor_target_grid(), steps=4, batch=4, seed=0)
    designer.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "policy.pt").exists()

    # from_pretrained on a local directory reloads the same policy: identical designs and scores.
    reloaded = Designer.from_pretrained(tmp_path)
    target = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
    original = designer.design(target, budget=16, tune_steps=1, threshold=0.0, seed=0)
    restored = reloaded.design(target, budget=16, tune_steps=1, threshold=0.0, seed=0)
    assert [grn.topology_hash() for grn, _ in original] == [grn.topology_hash() for grn, _ in restored]
    assert original.scores == restored.scores
