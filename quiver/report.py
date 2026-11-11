"""What a candidate circuit does, in quantities its parts justify.

A :class:`Report` carries only what the library's characterization can support. A quantity that cannot be
measured from these parts is *absent* from the report rather than present with a caveat, so nothing on
screen rests on a value that was assumed. :func:`measure` reads the quantities out of a simulation and
:func:`satisfaction` turns them into the acceptance score the search follows --- a plateau at ``1.0`` once
a candidate does what was asked, so the sampler spreads over the designs that work instead of climbing
toward a single one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from quiver.component import ComponentLibrary
from quiver.grn import GRN
from quiver.intent import Intent, Prefer, Responds
from quiver.objective.behavior import Behavior, fold_change, monotonicity
from quiver.objective.simulate import SimulationResult

# A device is called a switch once its two states separate by this much; below it the readout is not
# reliably distinguishable on a plate reader.
MIN_CONTRAST = 10.0

# The most expression allowed in the state that should be off, as a share of the on state.
MAX_BACKGROUND = 0.10


class Quantity(Enum):
    """A measurable property of a candidate, with the words it is reported in."""

    CONTRAST = "on vs off brightness"
    BACKGROUND = "leakiness"
    PART_COUNT = "parts"
    MEASURED_KINETICS = "kinetics from measurement"


def _format(quantity: Quantity, value: float) -> str:
    if quantity is Quantity.CONTRAST:
        return f"{value:.0f}x"
    if quantity is Quantity.BACKGROUND:
        return f"{value * 100:.0f}% of the on level"
    if quantity is Quantity.PART_COUNT:
        return f"{value:.0f}"
    return f"{value * 100:.0f}% measured, the rest assumed"


# Which measured quantity each preference ranks by, and whether more of it is better.
_RANK_BY: dict[Prefer, tuple[Quantity, bool]] = {
    Prefer.CONTRAST: (Quantity.CONTRAST, True),
    Prefer.LOW_BACKGROUND: (Quantity.BACKGROUND, False),
    Prefer.FEW_PARTS: (Quantity.PART_COUNT, False),
    Prefer.PREDICTABILITY: (Quantity.MEASURED_KINETICS, True),
}


@dataclass(frozen=True)
class Margin:
    """One requirement of a request, and how well a candidate meets it."""

    name: str
    required: float
    measured: float
    satisfaction: float

    @property
    def met(self) -> bool:
        return self.satisfaction >= 1.0

    def shortfall(self) -> str:
        """What fell short, in the words the requirement is stated in."""
        if self.name == "leakiness":
            return f"leakiness {self.measured * 100:.0f}% of the on level, needed {self.required * 100:.0f}% or less"
        if self.name == "on vs off brightness":
            return f"on vs off brightness {self.measured:.0f}x, needed {self.required:.0f}x or more"
        if self.name.startswith("responds"):
            return f"the response is not clean: {self.measured * 100:.0f}% of the dose steps {self.name[9:]}"
        return f"{self.name}: {self.measured * 100:.0f}% of what was needed"


@dataclass(frozen=True)
class Report:
    """What one candidate does. Quantities its parts cannot justify are absent."""

    asked: str
    measured: Mapping[Quantity, float]
    requirements: tuple[Margin, ...] = ()

    @property
    def score(self) -> float:
        """How well this candidate does what was asked."""
        return satisfaction(self.requirements)

    @property
    def meets(self) -> bool:
        """Whether every requirement is met."""
        return all(m.met for m in self.requirements)

    def unmet(self) -> tuple[Margin, ...]:
        """The requirements this candidate falls short of, worst first."""
        return tuple(sorted((m for m in self.requirements if not m.met), key=lambda m: m.satisfaction))

    def __getitem__(self, quantity: Quantity) -> float:
        return self.measured[quantity]

    def rank_by(self, prefer: Prefer) -> float:
        """A sort key for ``prefer``; higher is better. Unmeasurable preferences rank last."""
        quantity, more_is_better = _RANK_BY[prefer]
        if quantity not in self.measured:
            return float("-inf")
        value = self.measured[quantity]
        return value if more_is_better else -value

    def __str__(self) -> str:
        verdict = "does what you asked" if self.meets else "does not do what you asked"
        lines = [f"{verdict}:  {self.asked}", ""]
        width = max(len(q.value) for q in self.measured) if self.measured else 0
        for quantity, value in self.measured.items():
            lines.append(f"  {quantity.value:<{width}}  {_format(quantity, value)}")
        return "\n".join(lines)


def measure_response(result: SimulationResult, intent: Intent) -> dict[Quantity, float]:
    """The quantities readable from the assay alone, which is what the acceptance score is built on."""
    endpoints = result.endpoints(intent.reports.name)
    measured: dict[Quantity, float] = {}
    if len(endpoints) >= 2:
        high, low = max(endpoints), min(endpoints)
        measured[Quantity.CONTRAST] = fold_change(result, intent.reports.name)
        measured[Quantity.BACKGROUND] = low / high if high > 0.0 else 1.0
    return measured


def measure(
    grn: GRN, result: SimulationResult, intent: Intent, library: ComponentLibrary | None
) -> dict[Quantity, float]:
    """Everything this circuit's parts justify reporting: the assay quantities plus the circuit's own."""
    measured = measure_response(result, intent)
    measured[Quantity.PART_COUNT] = float(len(grn.parts_list()))
    if library is not None:
        measured[Quantity.MEASURED_KINETICS] = _measured_share(grn, library)
    return measured


def _measured_share(grn: GRN, library: ComponentLibrary) -> float:
    """The share of this circuit's kinetic values that come from a measurement rather than a default.

    Each grounded stage contributes its half-maximal point and its Hill coefficient; a stage whose part did
    not characterize them, or that is not grounded at all, contributes nothing. This is what
    :attr:`~quiver.intent.Prefer.PREDICTABILITY` ranks by, so a design resting on assumed values sorts last.
    """
    live = grn.effective()
    total = known = 0
    for node in grn.operators():
        if node.id not in live:
            continue
        total += 2
        if node.component_id is None:
            continue
        try:
            component = library.by_id(node.component_id)
        except KeyError:
            continue
        known += (component.K is not None) + (component.n is not None)
    return known / total if total else 0.0


def margins(measured: Mapping[Quantity, float], result: SimulationResult, intent: Intent) -> tuple[Margin, ...]:
    """Each requirement of the request, with how well this candidate meets it.

    A requirement's satisfaction plateaus at ``1.0`` once it is met, so candidates that all do the job look
    alike to the sampler and it spreads across them rather than concentrating on one.
    """
    direction = {Responds.UP: "up", Responds.DOWN: "down", Responds.PEAK: "any"}[intent.responds]
    endpoints = result.endpoints(intent.reports.name)
    found = [
        Margin(
            name=f"responds {direction}",
            required=1.0,
            measured=monotonicity(endpoints, direction),
            satisfaction=monotonicity(endpoints, direction),
        )
    ]

    contrast = measured.get(Quantity.CONTRAST, 1.0)
    found.append(
        Margin(
            name="on vs off brightness",
            required=MIN_CONTRAST,
            measured=contrast,
            satisfaction=min(1.0, contrast / MIN_CONTRAST),
        )
    )

    if Quantity.BACKGROUND in measured:
        background = measured[Quantity.BACKGROUND]
        found.append(
            Margin(
                name="leakiness",
                required=MAX_BACKGROUND,
                measured=background,
                satisfaction=1.0 if background <= MAX_BACKGROUND else MAX_BACKGROUND / max(background, 1e-9),
            )
        )
    return tuple(found)


def buildable(grn: GRN) -> Margin:
    """Whether every stage on the path to the reporter has a part assigned.

    Orthogonality limits how many stages a library can fill, so a circuit can be proposed with a stage
    left abstract. Such a stage has no part to order and no measured kinetics, so a design containing one
    cannot be built from the inventory it was designed against.
    """
    live = grn.effective()
    stages = [node for node in grn.operators() if node.id in live]
    grounded = sum(1 for node in stages if node.component_id is not None)
    share = grounded / len(stages) if stages else 0.0
    return Margin(name="every stage has a part", required=1.0, measured=share, satisfaction=share)


def satisfaction(found: Sequence[Margin]) -> float:
    """How well a candidate does what was asked, in ``(0, 1]``: the worst of its requirements."""
    return max(min((m.satisfaction for m in found), default=0.0), 1e-9)


def acceptance(intent: Intent) -> Behavior:
    """The acceptance score as a :data:`~quiver.objective.behavior.Behavior`, for the sampler to follow."""

    def detect(result: SimulationResult) -> float:
        return satisfaction(margins(measure_response(result, intent), result, intent))

    return detect
