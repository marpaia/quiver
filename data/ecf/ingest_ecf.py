"""Ingest the Rhodius et al. 2013 ECF sigma-factor library into ``ecf_activators.csv``.

ECF sigma factors activate their cognate promoters, so each functional sigma:promoter switch is an
``activator`` Component: its on-target *average fold induction* is the activation dynamic range. This script
reads the paper's supplementary data, pairs each sigma to its cognate promoter (matched by ECF group, taking
the promoter the sigma induces most), keeps the functional switches (fold >= ``MIN_FOLD``), and writes the
flat CSV :func:`quiver.load_ecf` consumes.

Source: Rhodius et al., "Design of orthogonal genetic switches based on a crosstalk map of sigmas, anti-sigmas,
and promoters", Mol Syst Biol 9:702 (2013), doi:10.1038/msb.2013.58 (CC BY). Supplement ``msb201358-s3.xlsx``,
sheet "S2.1 - Sigma orthogonality" (average fold induction, promoters x sigmas); species from
``msb201358-s2.xlsx`` sheet "S1.1". Only the dynamic range is used (a measured quantity); absolute levels are
left to the loader's nominal-OFF convention, which is why these are tagged ``quality="literature"``.

Obtain the supplements (deterministic, no login) via EuropePMC and run:

    curl -L -o supp.zip https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3817407/supplementaryFiles
    unzip supp.zip -d supp
    uv run --with openpyxl python ingest_ecf.py supp   # writes ecf_activators.csv alongside

Requires ``openpyxl`` (data-prep only; the shipped loader reads the CSV and needs no xlsx parser).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import openpyxl

# Minimum on-target fold induction for a sigma:promoter switch to be kept as a usable activator.
MIN_FOLD = 5.0


def _group(name: str) -> str | None:
    """The ECF group number of a sigma or promoter name (``ECF20_992`` / ``Pecf20_992`` -> ``20``)."""
    match = re.search(r"(\d+)", name.replace("Pecf", "").replace("ECF", ""))
    return match.group(1) if match else None


def _species(supp: Path) -> dict[str, str]:
    """Each ECF sigma's source species, from the S1.1 library sheet (best-effort metadata)."""
    path = supp / "msb201358-s2.xlsx"
    if not path.exists():
        return {}
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["S1.1 - ECF sigma library"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c) if c else "" for c in rows[0]]
    name_col = header.index("ECF sigma")
    species_col = next((i for i, h in enumerate(header) if h.upper() == "SPECIES"), None)
    out: dict[str, str] = {}
    for row in rows[1:]:
        if row[name_col] and species_col is not None and row[species_col]:
            out[str(row[name_col])] = str(row[species_col])
    return out


def activators(supp: Path) -> list[dict[str, object]]:
    """One row per functional ECF activator: sigma, cognate promoter, and on-target fold induction."""
    ws = openpyxl.load_workbook(supp / "msb201358-s3.xlsx", read_only=True, data_only=True)[
        "S2.1 - Sigma orthogonality"
    ]
    rows = list(ws.iter_rows(values_only=True))
    sigmas = [c for c in rows[1][2:] if c]  # sigma names: row index 1, columns from index 2
    sigma_col = {s: 2 + i for i, s in enumerate(sigmas)}
    promoter_rows = [r for r in rows[2:] if r and r[1]]  # promoter name in column index 1
    species = _species(supp)

    out: list[dict[str, object]] = []
    for sigma in sigmas:
        group = _group(str(sigma))
        best_fold, best_promoter = 0.0, None
        for row in promoter_rows:
            if _group(str(row[1])) != group:
                continue
            try:
                fold = float(row[sigma_col[sigma]])
            except (TypeError, ValueError):
                continue
            if fold > best_fold:
                best_fold, best_promoter = fold, str(row[1])
        if best_promoter is not None and best_fold >= MIN_FOLD:
            out.append(
                {
                    "name": sigma,
                    "regulator": sigma,  # the sigma factor is the orthogonality key
                    "role": "activator",
                    "family": "ECF",
                    "organism": species.get(str(sigma), ""),
                    "dynamic_range": round(best_fold, 3),
                    "cognate_promoter": best_promoter,
                    "source": "rhodius2013",
                    "quality": "literature",
                }
            )
    return out


def main() -> None:
    supp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("supp")
    rows = activators(supp)
    fields = [
        "name",
        "regulator",
        "role",
        "family",
        "organism",
        "dynamic_range",
        "cognate_promoter",
        "source",
        "quality",
    ]
    out_csv = Path(__file__).with_name("ecf_activators.csv")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    drs = [float(r["dynamic_range"]) for r in rows]  # type: ignore[arg-type]
    print(f"wrote {out_csv} ({len(rows)} activators, fold >= {MIN_FOLD})")
    if drs:
        print(f"dynamic range: {min(drs):.1f} .. {max(drs):.1f} (median {sorted(drs)[len(drs) // 2]:.1f})")


if __name__ == "__main__":
    main()
