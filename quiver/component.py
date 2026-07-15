"""The SBOL Component library: characterized genetic Components a design is assembled from.

A :class:`Component` is one characterized part in the SBOL3 sense --- a repressor gate or an input sensor with
measured Hill kinetics --- that the sampler selects to fill a circuit stage, so a delivered design is a real
bill-of-Components rather than an abstract role. :func:`load_cello` builds a :class:`ComponentLibrary` from the
Cello characterization corpus (``data/cello``); :func:`component_to_params` maps a Component's measured
``(ymax, ymin, K, n)`` onto the GRN/LOICA kinetic parameters of the operator it grounds, and
:func:`apply_component` stamps a Component's kinetics, name, and id onto a node.

The Cello corpus is repressor-only (its gates are NOR inverters) and its sensors carry no ``K``/``n``, so a
sensor grounds its expression levels from measurement and defaults its cooperativity; richer Components
(activators, sensor cooperativity) come from other sources (e.g. SynBioHub) plugged into the same library.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from quiver.generate.realizable import clamp_realizable
from quiver.grn import Node, ParamValue

ComponentRole = Literal["repressor", "sensor", "reporter"]

# The Cello corpus shipped in the repo; see data/cello/README.md.
_CELLO_PATH = Path(__file__).resolve().parent.parent / "data" / "cello" / "cello_kinetics.csv"

# Cooperativity defaulted for a sensor Component (the Cello input sensors carry no measured K/n).
_DEFAULT_SENSOR_K = 1.0
_DEFAULT_SENSOR_N = 2.0


@dataclass(frozen=True)
class Component:
    """One characterized SBOL Component: a repressor gate or input sensor with measured Hill kinetics.

    ``id`` is unique within a library (the sampler's vocabulary index); ``name`` is the display/part name
    carried onto the grounded node and into LOICA/SBOL export; ``regulator`` is the transcription factor (for
    a gate) or inducer small molecule (for a sensor), the orthogonality key --- two gates sharing a regulator
    cannot both appear in one circuit. ``K``/``n`` are ``None`` when the source did not characterize them.
    """

    id: str
    name: str
    role: ComponentRole
    regulator: str
    family: str
    organism: str
    ymax: float
    ymin: float
    K: float | None
    n: float | None
    dynamic_range: float
    source: str


@dataclass(frozen=True)
class ComponentLibrary:
    """A collection of characterized Components (an SBOL Collection) the sampler draws from."""

    components: tuple[Component, ...]

    def __len__(self) -> int:
        return len(self.components)

    def by_id(self, component_id: str) -> Component:
        for component in self.components:
            if component.id == component_id:
                return component
        raise KeyError(component_id)

    def repressors(self) -> tuple[Component, ...]:
        """The repressor-gate Components, ordered by id (the stable head index for stage selection)."""
        return tuple(sorted((c for c in self.components if c.role == "repressor"), key=lambda c: c.id))

    def sensors(self) -> tuple[Component, ...]:
        """The input-sensor Components, ordered by id."""
        return tuple(sorted((c for c in self.components if c.role == "sensor"), key=lambda c: c.id))

    def sensor_for(self, molecule: str) -> Component | None:
        """A sensor Component that responds to ``molecule`` (matched on its regulator/inducer, case- and
        suffix-insensitive so ``"aTc"`` matches an ``aTc_sensor``), preferring the widest dynamic range;
        ``None`` if the library has no sensor for it."""
        key = molecule.strip().lower()
        matches = [c for c in self.sensors() if key == c.regulator.strip().lower() or key in c.name.strip().lower()]
        return max(matches, key=lambda c: c.dynamic_range) if matches else None

    def regulators(self) -> frozenset[str]:
        """The distinct regulators (transcription factors / inducers) represented in the library."""
        return frozenset(c.regulator for c in self.components)


def load_cello(path: Path | None = None) -> ComponentLibrary:
    """Build a :class:`ComponentLibrary` from the Cello characterization corpus.

    Each ``gate`` row becomes a ``repressor`` Component and each ``sensor`` row a ``sensor`` Component; the id
    is ``"{ucf}:{name}"`` (unique per row, as a part name recurs across organisms and UCFs)."""
    rows = list(csv.DictReader((path or _CELLO_PATH).open()))
    components: list[Component] = []
    for row in rows:
        role: ComponentRole
        if row["kind"] == "gate":
            role = "repressor"
        elif row["kind"] == "sensor":
            role = "sensor"
        else:
            continue
        try:
            ymax, ymin, dr = float(row["ymax"]), float(row["ymin"]), float(row["dynamic_range"])
        except (KeyError, ValueError):
            continue
        components.append(
            Component(
                id=f"{row['ucf']}:{row['name']}",
                name=row["name"],
                role=role,
                regulator=row.get("regulator", ""),
                family=row.get("family", ""),
                organism=row.get("organism", ""),
                ymax=ymax,
                ymin=ymin,
                K=_optional_float(row.get("K")),
                n=_optional_float(row.get("n")),
                dynamic_range=dr,
                source=f"cello:{row['ucf']}",
            )
        )
    return ComponentLibrary(tuple(components))


def component_to_params(component: Component) -> dict[str, ParamValue]:
    """The GRN/LOICA kinetic parameters realizing ``component``, clamped to the realizable envelope.

    A repressor grounds a ``hill1`` operator with ``alpha = [ymax, ymin]`` (output high when its input is low
    --- the inverter sign convention); a sensor grounds a ``receiver`` with ``alpha = [ymin, ymax]``
    (activation by its inducer). A sensor's ``K``/``n`` default when the source did not characterize them.
    """
    if component.role == "repressor":
        params: dict[str, ParamValue] = {
            "alpha": [component.ymax, component.ymin],
            "K": component.K if component.K is not None else _DEFAULT_SENSOR_K,
            "n": component.n if component.n is not None else _DEFAULT_SENSOR_N,
        }
        return clamp_realizable("hill1", params)
    if component.role == "sensor":
        params = {
            "alpha": [component.ymin, component.ymax],
            "K": component.K if component.K is not None else _DEFAULT_SENSOR_K,
            "n": component.n if component.n is not None else _DEFAULT_SENSOR_N,
        }
        return clamp_realizable("receiver", params)
    raise ValueError(f"component role {component.role!r} has no operator kinetics")


def apply_component(node: Node, component: Component) -> Node:
    """Ground ``node`` in ``component``: stamp its measured kinetics, part name, and id onto the node."""
    return replace(node, name=component.name, params=component_to_params(component), component_id=component.id)


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
