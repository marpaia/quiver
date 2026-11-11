"""The parts you have: a characterized library, scoped to the conditions it was measured in.

A :class:`Parts` library is the inventory a design is built from --- the plasmids in a freezer, or a
published corpus used as an example --- together with the chassis and provenance that make its numbers
mean something. :class:`Inducer` and :class:`Reporter` are handles *into* a library, so a design call takes
objects rather than names and cannot silently mix two measurement conditions.

What a library was measured for also decides what can be asked of it. :meth:`Parts.can_design` reports the
behaviors its characterization supports and, for the rest, the measurement that would unlock them; a design
request that outruns the data raises :class:`NotCharacterized` rather than falling back on an assumed value.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from quiver.component import Component, ComponentLibrary, load_flat_csv

if TYPE_CHECKING:
    from quiver.intent import Capabilities

# The columns a library round-trips through, in the schema :func:`parts` reads.
_CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "role",
    "regulator",
    "detects",
    "family",
    "organism",
    "ymax",
    "ymin",
    "K",
    "n",
    "quality",
)


class Organism(Enum):
    """The host species a library's numbers were measured in."""

    E_COLI = "Escherichia coli"
    B_SUBTILIS = "Bacillus subtilis"
    S_CEREVISIAE = "Saccharomyces cerevisiae"
    OTHER = "other"


class Provenance(Enum):
    """Where a library's numbers come from, which sets how far they are trusted.

    ``IN_HOUSE`` numbers were measured by the user under the conditions they will build in. ``LITERATURE``
    numbers were read from a publication, so they carry the reporting conditions of that paper.
    """

    IN_HOUSE = auto()
    LITERATURE = auto()


@dataclass(frozen=True)
class Strain:
    """The chassis a library was characterized in. Two libraries measured in different strains are not
    comparable, because expression levels are relative to the host."""

    name: str
    organism: Organism = Organism.E_COLI

    def __str__(self) -> str:
        return self.name


class Measurement(Enum):
    """A measurement a behavior needs in order to be judged at all.

    Each value names a physical quantity rather than a column: ``COOPERATIVITY`` is the Hill coefficient
    that decides whether feedback can latch, ``DEGRADATION_RATE`` is the protein turnover that decides
    whether a feedback ring oscillates or settles, and ``DOSE_RESPONSE`` is a sensor's response measured
    against inducer concentration rather than against another promoter's output.
    """

    REGULATOR_STRENGTH = "how strongly each regulator drives or blocks its promoter"
    COOPERATIVITY = "the Hill coefficient of each regulator"
    ACTIVATOR = "at least one regulator that activates rather than represses"
    TWO_INPUT_GATE = "a promoter characterized with two regulator inputs"
    DEGRADATION_RATE = "how fast each regulator protein is degraded and diluted"
    DOSE_RESPONSE = "a sensor's response measured against inducer concentration"


# Both derive from LookupError rather than KeyError, whose str() is the repr of its argument and would
# print a multi-line message with escaped newlines.
class NoSensorFor(LookupError):
    """Nothing in the library responds to the requested molecule."""


class NoReporterCalled(LookupError):
    """The library has no reporter by that name."""


class DifferentConditions(ValueError):
    """Two libraries measured under different conditions cannot be combined or used in one circuit."""


class NotCharacterized(ValueError):
    """The requested behavior cannot be judged from what these parts were measured for."""


@dataclass(frozen=True)
class Inducer:
    """A molecule the library can detect, and the sensor part that detects it.

    Carries the library it came from, so a design built around it is grounded in one set of conditions.
    """

    molecule: str
    sensor: Component
    library: "Parts" = field(repr=False, compare=False)

    def __str__(self) -> str:
        return self.molecule


@dataclass(frozen=True)
class Reporter:
    """A readout gene. ``characterized=False`` marks one that is not measured yet --- usable as an output,
    but it contributes no kinetics of its own."""

    name: str
    library: "Parts | None" = field(default=None, repr=False, compare=False)
    characterized: bool = True

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Parts:
    """A characterized library, scoped to the conditions it was measured in.

    Declare one in code from :class:`~quiver.component.Component` parts, or read one from a CSV with
    :func:`parts`.
    """

    components: tuple[Component, ...]
    strain: Strain
    provenance: Provenance = Provenance.IN_HOUSE
    name: str = ""

    def __post_init__(self) -> None:
        # Accept any sequence of parts so a library can be declared from a list literal.
        if not isinstance(self.components, tuple):
            object.__setattr__(self, "components", tuple(self.components))

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)

    def __repr__(self) -> str:
        kinds = {
            "repressors": sum(1 for c in self.components if c.role == "repressor"),
            "activators": sum(1 for c in self.components if c.role == "activator"),
            "sensors": sum(1 for c in self.components if c.role == "sensor"),
            "reporters": sum(1 for c in self.components if c.role == "reporter"),
        }
        tally = ", ".join(f"{n} {label}" for label, n in kinds.items() if n)
        how = "in-house" if self.provenance is Provenance.IN_HOUSE else "from the literature"
        return f"{len(self.components)} parts, measured in {self.strain} ({how})\n  {tally}"

    @property
    def conditions(self) -> tuple[Strain, Provenance]:
        """The identity two libraries must share to be used together."""
        return (self.strain, self.provenance)

    def inducer(self, molecule: str) -> Inducer:
        """The handle for a molecule this library can detect.

        Matched on what a sensor *detects* --- the small molecule you add --- rather than on the
        transcription factor it acts through, which is a different thing and is the orthogonality key.
        """
        key = molecule.strip().lower()
        matches = [c for c in self.components if c.role == "sensor" and key in _detects_keys(c)]
        if not matches:
            detectable = ", ".join(self.detectable()) or "nothing"
            raise NoSensorFor(f"nothing in your library responds to {molecule}.\n  you can sense: {detectable}")
        return Inducer(molecule=molecule, sensor=max(matches, key=lambda c: c.dynamic_range), library=self)

    def reporter(self, name: str) -> Reporter:
        """The handle for a readout gene in this library."""
        key = name.strip().lower()
        for component in self.components:
            if component.role == "reporter" and component.name.strip().lower() == key:
                return Reporter(name=component.name, library=self)
        known = ", ".join(sorted(c.name for c in self.components if c.role == "reporter")) or "none"
        raise NoReporterCalled(
            f"your library has no reporter called {name}.\n"
            f"  reporters here: {known}\n"
            f"  for one you are about to clone in, use quiver.Reporter({name!r}, characterized=False)"
        )

    def detectable(self) -> tuple[str, ...]:
        """Every molecule this library has a sensor for."""
        found = {c.detects or c.regulator for c in self.components if c.role == "sensor"}
        return tuple(sorted(name for name in found if name))

    def has(self, measurement: Measurement) -> bool:
        """Whether the library carries a measurement, read from the parts rather than assumed."""
        if measurement is Measurement.REGULATOR_STRENGTH:
            return any(c.role in ("repressor", "activator") and c.dynamic_range > 1.0 for c in self.components)
        if measurement is Measurement.COOPERATIVITY:
            return any(c.role in ("repressor", "activator") and c.n is not None for c in self.components)
        if measurement is Measurement.ACTIVATOR:
            return any(c.role == "activator" for c in self.components)
        if measurement is Measurement.DOSE_RESPONSE:
            return any(c.role == "sensor" and c.K is not None and c.n is not None for c in self.components)
        # No corpus format in use records a two-input response surface or a protein turnover rate, so these
        # are unsatisfied until a library supplies them explicitly.
        return False

    def missing(self, *measurements: Measurement) -> tuple[Measurement, ...]:
        """Which of ``measurements`` this library does not carry."""
        return tuple(m for m in measurements if not self.has(m))

    def can_design(self) -> "Capabilities":
        """What this library's characterization supports, and what each gap would need.

        Imported at call time because the answer is a property of the *behaviors* a library can be asked
        for, which are defined alongside the request rather than alongside the parts.
        """
        from quiver.intent import capabilities_of

        return capabilities_of(self)

    def as_library(self) -> ComponentLibrary:
        """The underlying :class:`~quiver.component.ComponentLibrary` the sampler grounds in."""
        return ComponentLibrary(self.components, name=self.name or str(self.strain))

    def to_csv(self, path: Path) -> None:
        """Write this library back out in the schema :func:`parts` reads.

        A trained designer is only meaningful together with the parts it was trained on, so saving one
        saves its library too. Ids are written without the corpus prefix that :func:`parts` adds back, so a
        round trip reproduces the same Component ids and a design's part references stay resolvable.
        """
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_CSV_COLUMNS))
            writer.writeheader()
            for component in self.components:
                bare = component.id.split(":", 1)[-1] if ":" in component.id else component.id
                writer.writerow(
                    {
                        "id": bare,
                        "name": component.name,
                        "role": component.role,
                        "regulator": component.regulator,
                        "detects": component.detects or "",
                        "family": component.family,
                        "organism": component.organism,
                        "ymax": component.ymax,
                        "ymin": component.ymin,
                        "K": "" if component.K is None else component.K,
                        "n": "" if component.n is None else component.n,
                        "quality": component.quality,
                    }
                )


def _detects_keys(component: Component) -> frozenset[str]:
    """The names a sensor can be asked for by: what it detects, falling back to its regulator when the
    source did not distinguish the molecule from the transcription factor."""
    names = {component.detects or component.regulator, component.name.replace("_sensor", "")}
    return frozenset(name.strip().lower() for name in names if name)


def parts(
    source: str | Path,
    *,
    strain: Strain,
    provenance: Provenance = Provenance.IN_HOUSE,
    name: str = "",
) -> Parts:
    """Load a characterized library from a CSV, scoped to the conditions it was measured in.

    Columns follow the corpus schema in ``data/``: ``name`` is required; ``role`` is one of ``repressor``,
    ``activator``, ``sensor``, ``reporter``; ``regulator`` defaults to ``name``; levels are given as
    ``ymax``/``ymin`` or as ``dynamic_range``; ``K``, ``n``, ``family``, ``organism``, ``quality`` and
    ``source`` are optional and absent values stay absent rather than being defaulted.
    """
    path = Path(source)
    corpus = name or path.stem
    library = load_flat_csv(path, corpus=corpus, default_role="repressor")
    if not library.components:
        raise FileNotFoundError(f"no parts found in {path}")
    return Parts(components=library.components, strain=strain, provenance=provenance, name=corpus)


def combine(*libraries: Parts, name: str = "") -> Parts:
    """Combine libraries measured under the same conditions into one.

    Libraries from different chassis or different provenance are refused: expression levels are relative to
    the host and to the measurement setup, so numbers from different conditions do not compose.
    """
    if not libraries:
        raise ValueError("combine() needs at least one library")
    first = libraries[0]
    for other in libraries[1:]:
        if other.conditions != first.conditions:
            raise DifferentConditions(
                f"measured in {first.strain} vs {other.strain}. Numbers from different conditions do not "
                "compose. Cross-calibrate them first, or design from each library separately and compare "
                "the resulting circuits."
            )
    seen: set[str] = set()
    merged: list[Component] = []
    for library in libraries:
        for component in library.components:
            if component.id not in seen:
                seen.add(component.id)
                merged.append(component)
    return Parts(
        components=tuple(merged),
        strain=first.strain,
        provenance=first.provenance,
        name=name or "+".join(lib.name for lib in libraries),
    )
