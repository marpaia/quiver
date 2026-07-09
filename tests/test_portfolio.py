from __future__ import annotations

from pathlib import Path

import pytest

from quiver import Designer, Portfolio, design
from quiver.generate import example_inducible
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


def test_design_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        design(_behavior(), method="bogus")


@pytest.mark.sim
def test_design_returns_distinct_ranked_portfolio() -> None:
    pytest.importorskip("loica")
    portfolio = design(_behavior(), method="evolutionary", budget=40, tune_steps=1, threshold=0.0, seed=0)
    hashes = [grn.topology_hash() for grn, _ in portfolio]
    assert hashes, "expected at least one design"
    assert len(hashes) == len(set(hashes)), "portfolio must hold distinct topologies"
    assert portfolio.scores == sorted(portfolio.scores, reverse=True), "designs must be ranked best-first"


@pytest.mark.sim
def test_portfolio_networks_compile_to_loica() -> None:
    import loica as lc

    portfolio = design(_behavior(), method="evolutionary", budget=40, tune_steps=1, threshold=0.0, seed=0)
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
