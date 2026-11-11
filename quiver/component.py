"""The SBOL Component library: characterized genetic Components a design is assembled from.

A :class:`Component` is one characterized part in the SBOL3 sense --- a repressor gate, an activator, or an
input sensor with measured kinetics --- that the sampler selects to fill a circuit stage, so a delivered
design is a real bill-of-Components rather than an abstract role. :func:`component_to_params` maps a
Component's measured ``(ymax, ymin, K, n)`` onto the GRN/LOICA kinetic parameters of the operator it grounds,
and :func:`apply_component` stamps a Component's kinetics, name, and id onto a node.

Characterization is scarcer than identity, so no single corpus is both large and rich: :func:`load_cello`
supplies clean repressor and sensor kinetics but no activators; other corpora contribute the missing types.
Every Component carries its ``source`` and a ``quality`` tag, so a design can tell a measured repressor from
a literature-sourced activator. Combining corpora is :func:`quiver.parts.combine`, which refuses libraries
measured under different conditions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

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
    # The small molecule a sensor responds to, which is *not* its ``regulator``: a sensor's regulator is the
    # transcription factor it acts through (the orthogonality key), while this is what goes into the flask.
    # ``None`` for a non-sensor, or for a sensor whose source did not distinguish the two.
    detects: str | None = None

    @property
    def corpus(self) -> str:
        """The coarse source corpus (the ``source`` prefix, e.g. ``"cello"`` from ``"cello:Eco1C1G1T1..."``)."""
        return self.source.split(":", 1)[0]

    @classmethod
    def repressor(
        cls,
        name: str,
        *,
        ymax: float,
        ymin: float,
        K: float | None = None,
        n: float | None = None,
        regulator: str | None = None,
        quality: str = "measured",
        source: str = "",
    ) -> "Component":
        """A repressor gate: its transcription factor drives the promoter down."""
        return cls._gate("repressor", name, ymax, ymin, K, n, regulator, quality, source)

    @classmethod
    def activator(
        cls,
        name: str,
        *,
        ymax: float,
        ymin: float,
        K: float | None = None,
        n: float | None = None,
        regulator: str | None = None,
        quality: str = "measured",
        source: str = "",
    ) -> "Component":
        """An activator gate: its transcription factor drives the promoter up."""
        return cls._gate("activator", name, ymax, ymin, K, n, regulator, quality, source)

    @classmethod
    def sensor(
        cls,
        name: str,
        *,
        detects: str,
        ymax: float,
        ymin: float,
        K: float | None = None,
        n: float | None = None,
        regulator: str | None = None,
        quality: str = "measured",
        source: str = "",
    ) -> "Component":
        """An input sensor. ``detects`` is the molecule it responds to, which is distinct from the
        ``regulator`` it acts through."""
        component = cls._gate("sensor", name, ymax, ymin, K, n, regulator, quality, source)
        return replace(component, detects=detects)

    @classmethod
    def reporter(
        cls,
        name: str,
        *,
        ymax: float = 1.0,
        ymin: float = 0.0,
        quality: str = "measured",
        source: str = "",
    ) -> "Component":
        """A readout gene."""
        return cls._gate("reporter", name, ymax, ymin, None, None, None, quality, source)

    @classmethod
    def _gate(
        cls,
        role: ComponentRole,
        name: str,
        ymax: float,
        ymin: float,
        K: float | None,
        n: float | None,
        regulator: str | None,
        quality: str,
        source: str,
    ) -> "Component":
        """Fill in what a declared part implies: its id, its own name as the default regulator, and the
        dynamic range its levels give."""
        return cls(
            id=name,
            name=name,
            role=role,
            regulator=regulator if regulator is not None else name,
            family="",
            organism="",
            ymax=ymax,
            ymin=ymin,
            K=K,
            n=n,
            dynamic_range=ymax / ymin if ymin > 0.0 else float("inf"),
            source=source,
            quality=quality,
        )


@dataclass(frozen=True)
class ComponentLibrary:
    """A collection of characterized Components (an SBOL Collection) the sampler draws from.

    ``name`` identifies the corpus a Component came from, which its ``source`` and ``id`` carry through to a
    delivered design."""

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
        """A sensor Component that responds to ``molecule``, preferring the widest dynamic range; ``None``
        if the library has no sensor for it.

        Matched first on what the sensor ``detects`` --- the molecule itself --- then on its ``regulator``
        and name for sources that did not distinguish the molecule from the transcription factor it acts
        through (case- and suffix-insensitive, so ``"aTc"`` matches an ``aTc_sensor``).
        """
        key = molecule.strip().lower()
        matches = [
            c
            for c in self.sensors()
            if key == (c.detects or "").strip().lower()
            or key == c.regulator.strip().lower()
            or key in c.name.strip().lower()
        ]
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
    ``default_role``), ``family``, ``organism``, ``K``, ``n``, ``quality``, ``source``, ``detects`` (the
    molecule a sensor responds to, when that differs from its ``regulator``); and the levels, given
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
                detects=(row.get("detects") or "").strip() or None,
            )
        )
    return ComponentLibrary(tuple(components), name=corpus)


def load_ecf(path: Path | None = None) -> ComponentLibrary:
    """The ECF sigma-factor activator corpus (``data/ecf``), the activator counterpart to Cello's repressors.

    ECF sigma factors activate their target promoters, so these are ``activator`` Components; merged with
    :func:`load_cello` they let a design alternate activation and repression instead of forming repressor
    chains only. Empty until the corpus CSV is populated (see ``data/ecf/README.md``)."""
    return load_flat_csv(path or _ECF_PATH, corpus="ecf", default_role="activator")


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
    """Ground every ungrounded ``receiver`` in ``grn`` with a sensor Component responding to ``molecule``,
    and name the circuit's supplement for that molecule so the design reads as the assay it describes.
    A no-op if the library has no such sensor."""
    sensor = library.sensor_for(molecule)
    if sensor is None:
        return grn
    nodes = tuple(_grounded_input(node, sensor, molecule) for node in grn.nodes)
    return replace(grn, nodes=nodes)


def _grounded_input(node: Node, sensor: Component, molecule: str) -> Node:
    """A sensor stage grounded in ``sensor``, or a supplement renamed to the molecule being sensed."""
    if node.kind == "receiver" and node.component_id is None:
        return apply_component(node, sensor)
    if node.kind == "supplement":
        return replace(node, name=molecule)
    return node


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
