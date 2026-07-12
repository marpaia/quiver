"""Ingest Cello UCF response functions into a flat kinetic-parameter corpus.

Cello characterizes each repressor gate (and input sensor) with a Hill response function
``y = ymin + (ymax - ymin) / (1 + (x / K)**n)`` whose parameters live in the UCF ``models`` collection. This
script walks a checkout of CIDARLAB/Cello-UCF, extracts one row per gate or sensor across every organism
UCF, and writes a flat CSV plus a summary of the parameter ranges.

Provenance: https://github.com/CIDARLAB/Cello-UCF @ 222827d (v2 UCF files). Cello parameters are in RPU
(relative promoter units); ``K`` is in input-RPU.

Usage: python ingest_cello.py /path/to/Cello-UCF > /dev/null   # writes cello_kinetics.csv alongside
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HILL_KEYS = ("ymax", "ymin", "K", "n")


def _params(entry: dict) -> dict[str, float]:
    return {p["name"]: p["value"] for p in entry.get("parameters", []) if "value" in p}


def _models_by_name(docs: list[dict]) -> dict[str, dict]:
    return {d["name"]: d for d in docs if d.get("collection") == "models"}


def _organism(filename: str) -> str:
    """The organism prefix of a Cello file name (e.g. ``Eco1C1G1T1.UCF.json`` -> ``Eco``)."""
    stem = filename.split(".")[0]
    for i, ch in enumerate(stem):
        if ch.isdigit():
            return stem[:i]
    return stem[:3]


def rows_from_ucf(path: Path) -> list[dict]:
    """One row per gate: family/regulator metadata joined to its model's Hill parameters."""
    docs = json.loads(path.read_text())
    models = _models_by_name(docs)
    organism = _organism(path.name)
    out: list[dict] = []
    for gate in (d for d in docs if d.get("collection") == "gates"):
        model = models.get(gate.get("model", ""))
        if model is None:
            continue
        p = _params(model)
        if not all(k in p for k in HILL_KEYS):
            continue
        out.append(
            {
                "ucf": path.name,
                "organism": organism,
                "kind": "gate",
                "name": gate.get("name"),
                "family": gate.get("system"),
                "regulator": gate.get("regulator"),
                "gate_type": gate.get("gate_type"),
                "ymax": p["ymax"],
                "ymin": p["ymin"],
                "K": p["K"],
                "n": p["n"],
                "dynamic_range": p["ymax"] / p["ymin"] if p["ymin"] else float("nan"),
            }
        )
    return out


def rows_from_input(path: Path) -> list[dict]:
    """One row per input sensor. Sensors always carry ymax/ymin; K/n only when the model fits them."""
    docs = json.loads(path.read_text())
    models = _models_by_name(docs)
    out: list[dict] = []
    for sensor in (d for d in docs if d.get("collection") == "input_sensors"):
        model = models.get(sensor.get("model", ""))
        if model is None:
            continue
        p = _params(model)
        if "ymax" not in p or "ymin" not in p:
            continue
        out.append(
            {
                "ucf": path.name,
                "organism": _organism(path.name),
                "kind": "sensor",
                "name": sensor.get("name"),
                "family": "input_sensor",
                "regulator": sensor.get("name", "").replace("_sensor", ""),
                "gate_type": "SENSOR",
                "ymax": p["ymax"],
                "ymin": p["ymin"],
                "K": p.get("K", ""),
                "n": p.get("n", ""),
                "dynamic_range": p["ymax"] / p["ymin"] if p["ymin"] else float("nan"),
            }
        )
    return out


def main() -> None:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/cello-ucf-scout")
    rows: list[dict] = []
    for ucf in sorted((repo / "files/v2/ucf").rglob("*.UCF.json")):
        rows.extend(rows_from_ucf(ucf))
    for inp in sorted((repo / "files/v2/input").rglob("*.input.json")):
        rows.extend(rows_from_input(inp))

    out_csv = Path(__file__).with_name("cello_kinetics.csv")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gates = [r for r in rows if r["kind"] == "gate"]
    sensors = [r for r in rows if r["kind"] == "sensor"]

    def rng(vals: list[float]) -> str:
        vals = [v for v in vals if isinstance(v, (int, float))]
        return f"{min(vals):.3g} .. {max(vals):.3g} (median {statistics.median(vals):.3g})"

    print(f"wrote {out_csv} ({len(rows)} rows: {len(gates)} gates, {len(sensors)} sensors)")
    print("\norganisms:", sorted({r["organism"] for r in rows}))
    fams = sorted({r["family"] for r in gates})
    print(f"repressor families ({len(fams)}):", fams)
    print(f"distinct regulators: {len({r['regulator'] for r in gates})}")
    print("\nHill parameter ranges over gates:")
    print("  K            ", rng([r["K"] for r in gates]))
    print("  n            ", rng([r["n"] for r in gates]))
    print("  ymax         ", rng([r["ymax"] for r in gates]))
    print("  ymin         ", rng([r["ymin"] for r in gates]))
    print("  dynamic_range", rng([r["dynamic_range"] for r in gates]))
    if sensors:
        print("\nsensors dynamic_range:", rng([r["dynamic_range"] for r in sensors]))
        print("sensor names:", sorted({r["name"] for r in sensors}))


if __name__ == "__main__":
    main()
