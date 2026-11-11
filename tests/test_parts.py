"""The parts library: looking up what you have, and what its characterization lets you ask for."""

from __future__ import annotations

from pathlib import Path

import pytest

from quiver import (
    Component,
    DifferentConditions,
    Measurement,
    NoReporterCalled,
    NoSensorFor,
    Organism,
    Parts,
    Provenance,
    Strain,
    combine,
    parts,
)

_ROWS = [
    "id,name,role,regulator,detects,family,organism,ymax,ymin,K,n,quality",
    "p1,PhlF,repressor,PhlF,,TetR,Eco,3.9,0.01,0.03,4.0,measured",
    "p2,AmtR,repressor,AmtR,,TetR,Eco,3.8,0.06,0.07,1.6,measured",
    # A sensor's regulator is the transcription factor it acts through; what it detects is the molecule.
    "s1,AraC_sensor,sensor,AraC,arabinose,sensor,Eco,2.5,0.008,,,measured",
    "r1,GFP,reporter,GFP,,reporter,Eco,1.0,0.001,,,measured",
]


def _inventory(tmp_path: Path, rows: list[str] | None = None, **kwargs: object) -> object:
    path = tmp_path / "inventory.csv"
    path.write_text("\n".join(rows or _ROWS) + "\n")
    return parts(path, strain=Strain("MG1655", organism=Organism.E_COLI), **kwargs)  # type: ignore[arg-type]


def test_inducer_is_found_by_the_molecule_not_the_transcription_factor(tmp_path: Path) -> None:
    # A sensor is asked for by what you add to the flask, not by the protein it works through.
    inventory = _inventory(tmp_path)
    assert inventory.detectable() == ("arabinose",)
    assert inventory.inducer("arabinose").sensor.name == "AraC_sensor"


def test_missing_inducer_names_what_can_be_sensed(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(NoSensorFor, match="you can sense: arabinose"):
        inventory.inducer("IPTG")


def test_missing_reporter_points_at_the_uncharacterized_escape_hatch(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with pytest.raises(NoReporterCalled, match="characterized=False"):
        inventory.reporter("mCherry")


def test_measurements_are_read_from_the_parts_not_assumed(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    assert inventory.has(Measurement.REGULATOR_STRENGTH)
    assert inventory.has(Measurement.COOPERATIVITY)
    # The sensor carries no dose-response, and nothing in any corpus format reports protein turnover.
    assert not inventory.has(Measurement.DOSE_RESPONSE)
    assert not inventory.has(Measurement.DEGRADATION_RATE)
    assert inventory.missing(Measurement.DEGRADATION_RATE) == (Measurement.DEGRADATION_RATE,)


def test_capabilities_name_the_measurement_each_gap_needs(tmp_path: Path) -> None:
    text = str(_inventory(tmp_path).can_design())
    assert "responds UP / DOWN" in text
    assert "timing OSCILLATING" in text
    assert "degraded and diluted" in text, "a gap must say which measurement would close it"


def test_libraries_from_different_chassis_do_not_combine(tmp_path: Path) -> None:
    one = _inventory(tmp_path)
    other = parts(tmp_path / "inventory.csv", strain=Strain("DH10B"))
    with pytest.raises(DifferentConditions, match="MG1655 vs DH10B"):
        combine(one, other, name="mixed")


def test_combining_one_chassis_deduplicates_by_id(tmp_path: Path) -> None:
    one = _inventory(tmp_path)
    merged = combine(one, one, name="doubled")
    assert len(merged) == len(one)


def test_library_round_trips_through_csv(tmp_path: Path) -> None:
    # A saved designer saves its parts, so ids must survive the trip or part references stop resolving.
    original = _inventory(tmp_path, provenance=Provenance.IN_HOUSE)
    out = tmp_path / "saved.csv"
    original.to_csv(out)
    restored = parts(out, strain=Strain("MG1655"), name=original.name)
    assert [c.id for c in restored.components] == [c.id for c in original.components]
    assert [c.detects for c in restored.components] == [c.detects for c in original.components]
    assert [c.K for c in restored.components] == [c.K for c in original.components]


def test_a_library_can_be_declared_in_code() -> None:
    # Both paths build the same thing: parts declared as typed objects, or read from a CSV.
    inventory = Parts(
        [
            Component.repressor("PhlF", ymax=3.9, ymin=0.01, K=0.03, n=4.0),
            Component.sensor("AraC", detects="arabinose", ymax=2.5, ymin=0.008),
            Component.reporter("GFP"),
        ],
        strain=Strain("MG1655"),
    )
    assert len(inventory) == 3
    assert inventory.detectable() == ("arabinose",)
    assert inventory.inducer("arabinose").sensor.name == "AraC"
    assert inventory.reporter("GFP").name == "GFP"

    # A declared part fills in what its levels imply, and defaults its regulator to its own name.
    phlf = inventory.components[0]
    assert phlf.regulator == "PhlF" and abs(phlf.dynamic_range - 390.0) < 1e-9
    assert inventory.has(Measurement.COOPERATIVITY)
    # Only a sensor records a molecule it detects; a gate has none.
    assert phlf.detects is None and inventory.components[1].detects == "arabinose"
