"""The design call: what it refuses, what it delivers, and that training once serves many requests."""

from __future__ import annotations

from pathlib import Path

import pytest

from quiver import DifferentConditions, Logic, Prefer, Quantity, Reads, Responds, Strain, Timing, design, parts, train
from quiver.grn import GRN, Edge, Node
from quiver.intent import Contradiction, NotSupportedYet
from quiver.parts import NotCharacterized
from quiver.portfolio import Designer, Portfolio
from quiver.report import Margin, Report

_ROWS = [
    "id,name,role,regulator,detects,family,organism,ymax,ymin,K,n,quality",
    "p1,PhlF,repressor,PhlF,,TetR,Eco,3.9,0.01,0.03,4.0,measured",
    "p2,AmtR,repressor,AmtR,,TetR,Eco,3.8,0.06,0.07,1.6,measured",
    "p3,BetI,repressor,BetI,,TetR,Eco,3.8,0.07,0.41,2.4,measured",
    "p4,AmeR,repressor,AmeR,,TetR,Eco,3.8,0.2,0.09,1.4,measured",
    "s1,AraC_sensor,sensor,AraC,arabinose,sensor,Eco,2.5,0.008,,,measured",
    "s2,LacI_sensor,sensor,LacI,IPTG,sensor,Eco,2.8,0.0034,,,measured",
    "r1,GFP,reporter,GFP,,reporter,Eco,1.0,0.001,,,measured",
]


def _inventory(tmp_path: Path, strain: str = "MG1655") -> object:
    path = tmp_path / f"inventory-{strain}.csv"
    path.write_text("\n".join(_ROWS) + "\n")
    return parts(path, strain=Strain(strain))


def _report(meets: bool, measured: dict[Quantity, float] | None = None) -> Report:
    met = Margin(name="on vs off brightness", required=10.0, measured=30.0, satisfaction=1.0)
    short = Margin(name="leakiness", required=0.1, measured=0.4, satisfaction=0.25)
    return Report(asked="GFP comes on", measured=measured or {}, requirements=(met,) if meets else (met, short))


def test_portfolio_iterates_as_circuit_and_report() -> None:
    grn = GRN(nodes=(Node("out", "reporter", "GFP", {"signal_id": "gfp"}),), edges=())
    better = _report(True, {Quantity.CONTRAST: 30.0})
    worse = _report(False, {Quantity.CONTRAST: 2.0})
    portfolio = Portfolio(((grn, better), (grn, worse)), asked="GFP comes on")
    assert len(portfolio) == 2
    assert portfolio.best == (grn, better)
    assert [report.meets for _, report in portfolio] == [True, False]
    assert "2 candidates that do what you asked" in repr(portfolio)


def test_preference_reranks_without_designing_again() -> None:
    grn = GRN(nodes=(Node("out", "reporter", "GFP", {"signal_id": "gfp"}),), edges=())
    few = _report(True, {Quantity.PART_COUNT: 2.0})
    many = _report(True, {Quantity.PART_COUNT: 6.0})
    portfolio = Portfolio(((grn, many), (grn, few)))
    assert portfolio.best_for(Prefer.FEW_PARTS)[1] is few


def test_an_empty_batch_names_the_requirement_that_fell_short() -> None:
    grn = GRN(nodes=(Node("out", "reporter", "GFP", {"signal_id": "gfp"}),), edges=())
    empty = Portfolio((), asked="GFP comes on", closest=(grn, _report(False)))
    assert len(empty) == 0
    assert "nothing did what you asked" in repr(empty)
    assert "leakiness 40% of the on level, needed 10% or less" in repr(empty)


def test_build_sheet_lists_only_parts_that_reach_the_reporter() -> None:
    # A dead branch costs real reagents and changes nothing, so it must not appear on a build sheet.
    nodes = (
        Node("op", "hill1", "Live", {"alpha": [1.0, 0.0], "K": 1.0, "n": 2.0}, component_id="x:live"),
        Node("out", "reporter", "GFP", {"signal_id": "gfp"}),
        Node("dead", "hill1", "Dead", {"alpha": [1.0, 0.0], "K": 1.0, "n": 2.0}, component_id="x:dead"),
        Node("orphan", "regulator", "orphan", {}),
    )
    grn = GRN(nodes=nodes, edges=(Edge("op", "out"), Edge("dead", "orphan")))
    sheet = Portfolio(((grn, _report(True)),)).build_sheet()
    assert "Live" in sheet and "Dead" not in sheet


def test_contradictory_request_is_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(Contradiction, match="Pick one"):
        design(
            reports=inventory.reporter("GFP"),
            senses=inventory.inducer("arabinose"),
            memory=True,
            timing=Timing.OSCILLATING,
        )


def test_request_the_parts_cannot_judge_is_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(NotCharacterized, match="degraded and diluted"):
        design(reports=inventory.reporter("GFP"), timing=Timing.OSCILLATING)


def test_request_the_builder_cannot_construct_is_a_different_refusal(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(NotSupportedYet, match="feedforward"):
        design(reports=inventory.reporter("GFP"), senses=inventory.inducer("arabinose"), memory=True)


def test_logic_needs_more_than_one_signal(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(Contradiction, match="combines several signals"):
        design(
            reports=inventory.reporter("GFP"),
            senses=inventory.inducer("arabinose"),
            logic=Logic.BOTH,
        )


def test_parts_from_different_chassis_cannot_share_a_circuit(tmp_path: Path) -> None:
    mine, theirs = _inventory(tmp_path, "MG1655"), _inventory(tmp_path, "DH10B")
    with pytest.raises(DifferentConditions, match="MG1655 and DH10B"):
        design(reports=theirs.reporter("GFP"), senses=mine.inducer("arabinose"))


@pytest.mark.sim
def test_design_delivers_distinct_circuits_built_from_real_parts(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    portfolio = design(
        reports=inventory.reporter("GFP"),
        senses=inventory.inducer("arabinose"),
        steps=4,
        batch=2,
        explore=8,
        candidates=4,
        seed=0,
    )
    assert len(portfolio) >= 1
    layouts = [grn.topology_hash() for grn, _ in portfolio]
    assert len(layouts) == len(set(layouts)), "a batch holds distinct circuit layouts"
    for grn, report in portfolio:
        assert grn.parts_list(), "a delivered circuit is built from real parts"
        assert Quantity.CONTRAST in report.measured
        # These sensors carry no dose-response, so a threshold is never reported as if it were known.
        assert all("threshold" not in q.value for q in report.measured)


@pytest.mark.sim
def test_training_once_serves_several_signals_and_survives_a_round_trip(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    gfp = inventory.reporter("GFP")
    designer = train(inventory, reports=gfp, steps=4, batch=2, seed=0)

    for molecule in ("arabinose", "IPTG"):
        portfolio = designer.design(reports=gfp, senses=inventory.inducer(molecule), explore=6, candidates=3, seed=1)
        assert isinstance(portfolio, Portfolio)
        assert molecule in portfolio.asked

    designer.save_pretrained(tmp_path / "saved")
    reloaded = Designer.from_pretrained(tmp_path / "saved")
    # The library travels with the policy, or a reloaded designer could not ground anything.
    assert [c.id for c in reloaded.library.components] == [c.id for c in inventory.components]

    ara = inventory.inducer("arabinose")
    before = designer.design(reports=gfp, senses=ara, explore=6, candidates=3, seed=2)
    after = reloaded.design(
        reports=reloaded.library.reporter("GFP"),
        senses=reloaded.library.inducer("arabinose"),
        explore=6,
        candidates=3,
        seed=2,
    )
    assert [g.topology_hash() for g, _ in before] == [g.topology_hash() for g, _ in after]


@pytest.mark.sim
def test_both_policy_representations_design_through_one_call(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    for reads in (Reads.SUMMARY, Reads.GRAPH):
        portfolio = design(
            reports=inventory.reporter("GFP"),
            senses=inventory.inducer("arabinose"),
            responds=Responds.UP,
            reads=reads,
            steps=3,
            batch=2,
            explore=6,
            candidates=3,
            seed=0,
        )
        assert len(portfolio) >= 1
