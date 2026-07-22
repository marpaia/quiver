"""The SBOL Component library: characterized genetic Components a design is assembled from.

A :class:`Component` is one characterized part in the SBOL3 sense --- a repressor gate, an activator, or an
input sensor with measured kinetics --- that the sampler selects to fill a circuit stage, so a delivered
design is a real bill-of-Components rather than an abstract role. :func:`component_to_params` maps a
Component's measured ``(ymax, ymin, K, n)`` onto the GRN/LOICA kinetic parameters of the operator it grounds,
and :func:`apply_component` stamps a Component's kinetics, name, and id onto a node.

Characterization is scarcer than identity, so no single corpus is both large and rich: :func:`load_cello`
supplies clean repressor and sensor kinetics but no activators; other corpora contribute the missing types.
:func:`merge` aggregates several corpora into one library, and every Component carries its ``source`` and a
``quality`` tag so a benchmark can tell a measured repressor from a literature-sourced activator and filter
accordingly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from quiver.generate.realizable import clamp_realizable
from quiver.grn import GRN, Node, ParamValue

# A Component's regulatory role. Repressors and activators are *gates* (a regulator drives a promoter);
# sensors are the inducible input layer; reporters are outputs.
ComponentRole = Literal["repressor", "activator", "sensor", "reporter"]

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
    # How the kinetics were obtained, for cross-corpus provenance: "measured" (a fitted response function),
    # "fitted" (fit here from induction curves), or "literature" (read from a paper's reported values).
    quality: str = "measured"

    @property
    def corpus(self) -> str:
        """The coarse source corpus (the ``source`` prefix, e.g. ``"cello"`` from ``"cello:Eco1C1G1T1..."``)."""
        return self.source.split(":", 1)[0]


@dataclass(frozen=True)
class ComponentLibrary:
    """A collection of characterized Components (an SBOL Collection) the sampler draws from.

    ``name`` identifies the source so a trained :class:`~quiver.portfolio.Designer` can record which library it
    was grounded in and reload it (see :func:`library_from_name`)."""

    components: tuple[Component, ...]
    name: str = ""

    def __len__(self) -> int:
        return len(self.components)

    def by_id(self, component_id: str) -> Component:
        for component in self.components:
            if component.id == component_id:
                return component
        raise KeyError(component_id)

    def repressors(self) -> tuple[Component, ...]:
        """The repressor-gate Components, ordered by id."""
        return tuple(sorted((c for c in self.components if c.role == "repressor"), key=lambda c: c.id))

    def activators(self) -> tuple[Component, ...]:
        """The activator-gate Components, ordered by id."""
        return tuple(sorted((c for c in self.components if c.role == "activator"), key=lambda c: c.id))

    def gates(self) -> tuple[Component, ...]:
        """The gate Components a promoter stage can be grounded in --- repressors and activators together,
        ordered by id (the stable head index the sampler's Component selection addresses). A mixed library
        lets the policy choose activation or repression per stage, not just repressor chains."""
        return tuple(sorted((c for c in self.components if c.role in ("repressor", "activator")), key=lambda c: c.id))

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
    return ComponentLibrary(tuple(components), name="cello")


# The corpus of ECF sigma-factor activators; see data/ecf/README.md for the schema and how to populate it.
_ECF_PATH = Path(__file__).resolve().parent.parent / "data" / "ecf" / "ecf_activators.csv"

# A nominal OFF expression level (RPU) for a literature component reported only by its fold-change, so a
# dynamic range can be turned into (ymin, ymax). Only the dynamic range is a real measurement; the absolute
# level is a placeholder, which is why such components are tagged quality="literature" and the realizable
# clamp (dimensionless quantities only) is what the objective ultimately trusts.
_NOMINAL_OFF = 0.05


def load_flat_csv(
    path: Path,
    *,
    corpus: str,
    default_role: ComponentRole = "activator",
    quality: str = "literature",
) -> ComponentLibrary:
    """Load a characterized-component corpus from a flat CSV, the generic ingestion path for literature and
    third-party corpora (the reusable half of multi-corpus aggregation).

    Columns: ``name`` (required); ``regulator`` (defaults to ``name``); optional ``role`` (defaults to
    ``default_role``), ``family``, ``organism``, ``K``, ``n``, ``quality``, ``source``; and the levels, given
    either as ``ymax`` + ``ymin`` or as ``dynamic_range`` alone (then ``ymin`` is a nominal OFF level and
    ``ymax = ymin * dynamic_range``). Returns an empty library if ``path`` is absent, so a corpus that has not
    been populated yet is a no-op rather than an error."""
    if not path.exists():
        return ComponentLibrary((), name=corpus)
    components: list[Component] = []
    for row in csv.DictReader(path.open()):
        name = (row.get("name") or "").strip()
        if not name:
            continue
        ymax_raw, ymin_raw = _optional_float(row.get("ymax")), _optional_float(row.get("ymin"))
        dr_raw = _optional_float(row.get("dynamic_range"))
        if ymax_raw is not None and ymin_raw is not None and ymin_raw > 0:
            ymin, ymax, dr = ymin_raw, ymax_raw, ymax_raw / ymin_raw
        elif dr_raw is not None and dr_raw > 0:
            ymin, ymax, dr = _NOMINAL_OFF, _NOMINAL_OFF * dr_raw, dr_raw
        else:
            continue
        role_raw = (row.get("role") or default_role).strip()
        valid_role = role_raw in ("repressor", "activator", "sensor", "reporter")
        role: ComponentRole = role_raw if valid_role else default_role  # type: ignore[assignment]
        ref = (row.get("source") or "").strip()
        components.append(
            Component(
                id=f"{corpus}:{row.get('id') or name}",
                name=name,
                role=role,
                regulator=(row.get("regulator") or name).strip(),
                family=(row.get("family") or "").strip(),
                organism=(row.get("organism") or "").strip(),
                ymax=ymax,
                ymin=ymin,
                K=_optional_float(row.get("K")),
                n=_optional_float(row.get("n")),
                dynamic_range=dr,
                source=f"{corpus}:{ref}" if ref else corpus,
                quality=(row.get("quality") or quality).strip(),
            )
        )
    return ComponentLibrary(tuple(components), name=corpus)


def load_ecf(path: Path | None = None) -> ComponentLibrary:
    """The ECF sigma-factor activator corpus (``data/ecf``), the activator counterpart to Cello's repressors.

    ECF sigma factors activate their target promoters, so these are ``activator`` Components; merged with
    :func:`load_cello` they let a design alternate activation and repression instead of forming repressor
    chains only. Empty until the corpus CSV is populated (see ``data/ecf/README.md``)."""
    return load_flat_csv(path or _ECF_PATH, corpus="ecf", default_role="activator")


def merge(*libraries: ComponentLibrary, name: str) -> ComponentLibrary:
    """Aggregate several corpora into one :class:`ComponentLibrary`, de-duplicating by Component id (first
    occurrence wins). This is how a large, type-rich library is built when no single corpus has everything:
    e.g. ``merge(load_cello(), load_ecf(), name="cello+ecf")`` for repressors, sensors, and activators."""
    seen: set[str] = set()
    components: list[Component] = []
    for library in libraries:
        for component in library.components:
            if component.id not in seen:
                seen.add(component.id)
                components.append(component)
    return ComponentLibrary(tuple(components), name=name)


# The named libraries a saved Designer can reload by name. Composite names (``"a+b"``) merge their parts.
_LIBRARIES: dict[str, Callable[[], ComponentLibrary]] = {"cello": load_cello, "ecf": load_ecf}


def library_from_name(name: str) -> ComponentLibrary | None:
    """Rebuild a named :class:`ComponentLibrary` (as recorded by a saved Designer), or ``None`` if the name
    is empty or unknown. A ``"+"``-joined name merges its named parts (e.g. ``"cello+ecf"``)."""
    if not name:
        return None
    parts = name.split("+")
    loaders = [_LIBRARIES.get(part) for part in parts]
    if any(loader is None for loader in loaders):
        return None
    libraries = [loader() for loader in loaders if loader is not None]
    if len(libraries) == 1:
        return libraries[0]
    return merge(*libraries, name=name)


def component_to_params(component: Component) -> dict[str, ParamValue]:
    """The GRN/LOICA kinetic parameters realizing ``component``, clamped to the realizable envelope.

    A repressor grounds a ``hill1`` with ``alpha = [ymax, ymin]`` (output high when its input is low, the
    inverter sign); an activator grounds a ``hill1`` with ``alpha = [ymin, ymax]`` (output high when its input
    is high); a sensor grounds a ``receiver`` with ``alpha = [ymin, ymax]`` (activation by its inducer).
    ``K``/``n`` default when the source did not characterize them.
    """
    k = component.K if component.K is not None else _DEFAULT_SENSOR_K
    n = component.n if component.n is not None else _DEFAULT_SENSOR_N
    if component.role == "repressor":
        return clamp_realizable("hill1", {"alpha": [component.ymax, component.ymin], "K": k, "n": n})
    if component.role == "activator":
        return clamp_realizable("hill1", {"alpha": [component.ymin, component.ymax], "K": k, "n": n})
    if component.role == "sensor":
        return clamp_realizable("receiver", {"alpha": [component.ymin, component.ymax], "K": k, "n": n})
    raise ValueError(f"component role {component.role!r} has no operator kinetics")


def apply_component(node: Node, component: Component) -> Node:
    """Ground ``node`` in ``component``: stamp its measured kinetics, part name, and id onto the node."""
    return replace(node, name=component.name, params=component_to_params(component), component_id=component.id)


def used_regulators(grn: GRN, library: ComponentLibrary) -> set[str]:
    """The transcription factors / inducers already committed in ``grn`` (from its grounded nodes)."""
    used: set[str] = set()
    for node in grn.nodes:
        if node.component_id is not None:
            try:
                used.add(library.by_id(node.component_id).regulator)
            except KeyError:
                continue
    return used


def gate_availability(grn: GRN, library: ComponentLibrary) -> list[bool]:
    """A per-gate availability flag aligned to ``library.gates()`` (repressors and activators): ``False`` for a
    gate whose regulator is already used in ``grn`` (orthogonality --- no transcription factor twice in one
    circuit). This is the mask the sampler's Component head applies when grounding a promoter stage."""
    used = used_regulators(grn, library)
    return [component.regulator not in used for component in library.gates()]


def ungrounded_operator_ids(grn: GRN, kind: str = "hill1") -> list[str]:
    """The ids of operators of ``kind`` not yet grounded in a Component, in node order."""
    return [node.id for node in grn.nodes if node.kind == kind and node.component_id is None]


def ground_operator(grn: GRN, node_id: str, component: Component) -> GRN:
    """A copy of ``grn`` with the node ``node_id`` grounded in ``component``."""
    nodes = tuple(apply_component(node, component) if node.id == node_id else node for node in grn.nodes)
    return replace(grn, nodes=nodes)


def ground_sensor(grn: GRN, library: ComponentLibrary, molecule: str) -> GRN:
    """Ground every ungrounded ``receiver`` in ``grn`` with a sensor Component responding to ``molecule``;
    a no-op if the library has no such sensor."""
    sensor = library.sensor_for(molecule)
    if sensor is None:
        return grn
    nodes = tuple(
        apply_component(node, sensor) if node.kind == "receiver" and node.component_id is None else node
        for node in grn.nodes
    )
    return replace(grn, nodes=nodes)


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
