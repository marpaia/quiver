"""The design request: what it describes, and the three ways it can be refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from quiver import Logic, Responds, Strain, Timing, parts
from quiver.intent import (
    INTENT_FEATURES,
    Contradiction,
    Intent,
    NotSupportedYet,
    check_characterized,
    check_consistent,
    check_supported,
)
from quiver.parts import NotCharacterized

_ROWS = [
    "id,name,role,regulator,detects,family,organism,ymax,ymin,K,n,quality",
    "p1,PhlF,repressor,PhlF,,TetR,Eco,3.9,0.01,0.03,4.0,measured",
    "s1,AraC_sensor,sensor,AraC,arabinose,sensor,Eco,2.5,0.008,,,measured",
    "s2,LacI_sensor,sensor,LacI,IPTG,sensor,Eco,2.8,0.0034,,,measured",
    "r1,GFP,reporter,GFP,,reporter,Eco,1.0,0.001,,,measured",
]


def _intent(tmp_path: Path, **kwargs: object) -> Intent:
    path = tmp_path / "inventory.csv"
    path.write_text("\n".join(_ROWS) + "\n")
    inventory = parts(path, strain=Strain("MG1655"))
    fields: dict[str, object] = {
        "reports": inventory.reporter("GFP"),
        "senses": (inventory.inducer("arabinose"),),
    }
    fields.update(kwargs)
    return Intent(**fields)  # type: ignore[arg-type]


def test_describe_restates_the_request_in_plain_language(tmp_path: Path) -> None:
    assert _intent(tmp_path).describe() == "GFP comes on when the cell sees arabinose"
    assert "goes off" in _intent(tmp_path, responds=Responds.DOWN).describe()
    assert "washed out" in _intent(tmp_path, memory=True).describe()


def test_contradictory_choices_are_refused(tmp_path: Path) -> None:
    with pytest.raises(Contradiction, match="Pick one"):
        check_consistent(_intent(tmp_path, memory=True, timing=Timing.OSCILLATING))
    with pytest.raises(Contradiction, match="takes no input signal"):
        check_consistent(_intent(tmp_path, timing=Timing.OSCILLATING))
    with pytest.raises(Contradiction, match="combines several signals"):
        check_consistent(_intent(tmp_path, logic=Logic.BOTH))


def test_a_behavior_the_parts_cannot_judge_is_refused_with_the_measurement(tmp_path: Path) -> None:
    # No corpus format reports protein turnover, so oscillation cannot be judged from these parts at all.
    with pytest.raises(NotCharacterized, match="degraded and diluted"):
        check_characterized(_intent(tmp_path, senses=(), timing=Timing.OSCILLATING))


def test_a_behavior_the_builder_cannot_construct_is_a_different_refusal(tmp_path: Path) -> None:
    # Cooperativity *is* measured here, so a latch is answerable; it is the builder that cannot reach one.
    intent = _intent(tmp_path, memory=True)
    check_characterized(intent)
    with pytest.raises(NotSupportedYet, match="feedforward"):
        check_supported(intent)


def test_features_encode_the_axes_and_the_sensors_measured_properties(tmp_path: Path) -> None:
    up = _intent(tmp_path).features()
    down = _intent(tmp_path, responds=Responds.DOWN).features()
    assert len(up) == INTENT_FEATURES and len(down) == INTENT_FEATURES
    assert up != down, "the behavior axes must move the conditioning vector"

    path = tmp_path / "inventory.csv"
    inventory = parts(path, strain=Strain("MG1655"))
    other = _intent(tmp_path, senses=(inventory.inducer("IPTG"),)).features()
    assert up != other, "two sensors with different measured properties must encode differently"

    # An unmeasured property encodes as zero, which is distinguishable from a measured value.
    assert up[-2:] == [0.0, 0.0], "these sensors carry no K or n, and that must show"
