"""What you want the cell to do, as four orthogonal choices.

``responds``, ``logic``, ``memory`` and ``timing`` compose, so the askable behaviors are a small structured
space rather than a fixed menu of named devices: "remembers whether it saw both signals" is a point in it.
None of them is a number --- a quantity is something this library measures and reports, never something a
design request has to supply.

Each choice also names the measurements it can only be judged by. :func:`check_characterized` refuses a
request whose parts cannot answer it, and :func:`check_supported` refuses one this library cannot yet build,
which are different failures and say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import log10

from quiver.component import Component
from quiver.parts import Inducer, Measurement, NotCharacterized, Parts, Reporter


class Responds(Enum):
    """How the readout moves with the signal."""

    UP = auto()
    DOWN = auto()
    PEAK = auto()


class Logic(Enum):
    """How several signals combine."""

    PRESENT = auto()
    BOTH = auto()
    EITHER = auto()
    NEITHER = auto()
    ONE_ONLY = auto()


class Timing(Enum):
    """How the readout behaves over time once the signal settles."""

    SUSTAINED = auto()
    TRANSIENT = auto()
    OSCILLATING = auto()


class Prefer(Enum):
    """A direction to favor among candidates, when the first batch is not good enough.

    A preference ranks candidates this library has already measured, so unlike a numeric bound it can never
    be unanswerable.
    """

    CONTRAST = auto()
    LOW_BACKGROUND = auto()
    FEW_PARTS = auto()
    PREDICTABILITY = auto()


@dataclass(frozen=True)
class Capabilities:
    """What a library's characterization does and does not support, and the measurement that would unlock
    each unsupported behavior. Printing it gives the worklist."""

    available: tuple[tuple[str, str], ...]
    unavailable: tuple[tuple[str, Measurement], ...]

    def __str__(self) -> str:
        lines: list[str] = []
        if self.available:
            lines.append("you can ask for:")
            width = max(len(name) for name, _ in self.available)
            lines += [f"  {name:<{width}}  {why}" for name, why in self.available]
        if self.unavailable:
            lines.append("")
            lines.append("you cannot ask for:")
            width = max(len(name) for name, _ in self.unavailable)
            lines += [f"  {name:<{width}}  needs {need.value}" for name, need in self.unavailable]
        return "\n".join(lines)


class Reads(Enum):
    """How the sampler's policy sees a part-built circuit while it assembles one.

    ``SUMMARY`` reads a few numbers describing the circuit so far; ``GRAPH`` reads it as a graph, with each
    part's measured kinetics and signed wiring passed between neighbors. This is a representation choice
    within one strategy, not a different strategy, and which one wins depends on the regime.
    """

    SUMMARY = auto()
    GRAPH = auto()


class Contradiction(ValueError):
    """Two choices that cannot hold at once."""


class NotSupportedYet(NotImplementedError):
    """The request is answerable from these parts, but this library cannot construct it yet.

    Distinct from :class:`~quiver.parts.NotCharacterized`, which means the parts cannot answer it at all.
    """


@dataclass(frozen=True)
class Intent:
    """A resolved design request."""

    reports: Reporter
    senses: tuple[Inducer, ...]
    responds: Responds = Responds.UP
    logic: Logic = Logic.PRESENT
    memory: bool = False
    timing: Timing = Timing.SUSTAINED

    @property
    def library(self) -> Parts:
        """The one library this design is grounded in."""
        if self.senses:
            return self.senses[0].library
        if self.reports.library is not None:
            return self.reports.library
        raise ValueError("a design needs at least one part from a library")

    def describe(self) -> str:
        """The request restated in plain language, for the report's first line."""
        signals = " and ".join(str(i) for i in self.senses)
        if self.timing is Timing.OSCILLATING:
            return f"{self.reports} cycles on its own"
        direction = {Responds.UP: "comes on", Responds.DOWN: "goes off", Responds.PEAK: "peaks"}[self.responds]
        condition = {
            Logic.PRESENT: f"when the cell sees {signals}",
            Logic.BOTH: f"only when both {signals} are present",
            Logic.EITHER: f"when either {signals} is present",
            Logic.NEITHER: f"only when neither {signals} is present",
            Logic.ONE_ONLY: f"only when exactly one of {signals} is present",
        }[self.logic]
        said = f"{self.reports} {direction} {condition}"
        if self.memory:
            said += ", and stays that way after the signal is washed out"
        if self.timing is Timing.TRANSIENT:
            said += ", then settles back"
        return said

    def features(self) -> list[float]:
        """The policy's encoding of this request, for a conditioned (amortized) sampler.

        The behavior axes are one-hot, and the signal is encoded by its sensor's *measured properties*
        rather than its identity, so one trained designer can design for an inducer it never saw: a new
        sensor with a similar response is a nearby point rather than an unknown label.
        """
        encoding = [
            *(1.0 if self.responds is value else 0.0 for value in Responds),
            *(1.0 if self.logic is value else 0.0 for value in Logic),
            1.0 if self.memory else 0.0,
            *(1.0 if self.timing is value else 0.0 for value in Timing),
        ]
        return encoding + _sensor_features(self.senses[0].sensor if self.senses else None)

    def requires(self) -> tuple[Measurement, ...]:
        """The measurements this request can only be judged by."""
        needed = [Measurement.REGULATOR_STRENGTH]
        if self.memory:
            needed.append(Measurement.COOPERATIVITY)
        if self.logic is not Logic.PRESENT:
            needed.append(Measurement.TWO_INPUT_GATE)
        if self.timing is Timing.OSCILLATING:
            needed += [Measurement.COOPERATIVITY, Measurement.DEGRADATION_RATE]
        if self.timing is Timing.TRANSIENT:
            needed.append(Measurement.DEGRADATION_RATE)
        return tuple(dict.fromkeys(needed))


# How many numbers :meth:`Intent.features` produces: the behavior axes one-hot, plus the sensor's
# measured properties. A saved designer records it so a reloaded policy is checked against its training.
INTENT_FEATURES = len(Responds) + len(Logic) + 1 + len(Timing) + 3


def _sensor_features(sensor: Component | None) -> list[float]:
    """A sensor's measured properties, normalized: its dynamic range, half-maximal point, and
    cooperativity. A property the source did not measure encodes as zero, which is distinguishable from a
    measured value and keeps the width fixed."""
    if sensor is None:
        return [0.0, 0.0, 0.0]
    return [
        log10(max(sensor.dynamic_range, 1.0)) / 3.0,
        log10(sensor.K) / 3.0 if sensor.K is not None and sensor.K > 0.0 else 0.0,
        sensor.n / 5.0 if sensor.n is not None else 0.0,
    ]


# Why each gap matters, so a refusal explains the physics rather than naming a missing column.
_WHY: dict[Measurement, str] = {
    Measurement.REGULATOR_STRENGTH: (
        "how far a regulator moves its promoter is what decides whether the readout separates at all"
    ),
    Measurement.COOPERATIVITY: (
        "whether feedback can hold a state depends on how sharply each regulator responds to its input"
    ),
    Measurement.DEGRADATION_RATE: (
        "whether a feedback loop oscillates or just settles is set by how fast its proteins are degraded "
        "and diluted"
    ),
    Measurement.TWO_INPUT_GATE: (
        "combining two signals at one promoter needs that promoter measured against both of them"
    ),
    Measurement.ACTIVATOR: "turning a promoter on, rather than only off, needs a regulator that activates",
    Measurement.DOSE_RESPONSE: (
        "a threshold in concentration units needs the sensor measured against inducer concentration, not "
        "against another promoter's output"
    ),
}

_UNLOCK: dict[Measurement, str] = {
    Measurement.REGULATOR_STRENGTH: "measure the on and off levels of at least one regulator",
    Measurement.COOPERATIVITY: "fit a Hill coefficient for at least one regulator you would build with",
    Measurement.DEGRADATION_RATE: "measure protein degradation for at least one regulator you would build with",
    Measurement.TWO_INPUT_GATE: "characterize one promoter against two regulators together",
    Measurement.ACTIVATOR: "add an activating regulator to your library",
    Measurement.DOSE_RESPONSE: "run a dose-response on your sensors and record K and n",
}


def check_consistent(intent: Intent) -> None:
    """Refuse choices that cannot hold at once."""
    if intent.memory and intent.timing is Timing.OSCILLATING:
        raise Contradiction("a latch holds one state and an oscillator never settles. Pick one.")
    if intent.memory and intent.timing is Timing.TRANSIENT:
        raise Contradiction("a latch holds its state and a transient pulse returns to baseline. Pick one.")
    if intent.timing is Timing.OSCILLATING and intent.senses:
        raise Contradiction(
            "an oscillator runs on its own, so it takes no input signal. Drop senses=, or ask for "
            "timing=Timing.SUSTAINED if you want the signal to drive it."
        )
    if intent.timing is not Timing.OSCILLATING and not intent.senses:
        raise Contradiction("a circuit that responds to a signal needs senses= to say which signal.")
    if intent.logic is Logic.PRESENT and len(intent.senses) > 1:
        raise Contradiction(
            f"{len(intent.senses)} signals were given but logic is Logic.PRESENT, which is for one signal. "
            "Say how they combine, e.g. logic=Logic.BOTH."
        )
    if intent.logic is not Logic.PRESENT and len(intent.senses) < 2:
        raise Contradiction(f"logic={intent.logic.name} combines several signals, but only one was given.")


def check_characterized(intent: Intent) -> None:
    """Refuse a request these parts cannot be judged against, naming the measurement that would unlock it."""
    library = intent.library
    missing = library.missing(*intent.requires())
    if not missing:
        return
    gap = missing[0]
    raise NotCharacterized(
        f"{_WHY[gap]}, and none of your {len(library)} parts report it.\n"
        f"  to unlock this: {_UNLOCK[gap]}."
    )


def check_supported(intent: Intent) -> None:
    """Refuse a request this library cannot construct yet.

    These are gaps in the generator, not in the parts, and are reported separately so a user is never told
    their data is insufficient when the tool is what is missing.
    """
    if intent.memory:
        raise NotSupportedYet(
            "holding a state needs a feedback loop, and the circuit builder only assembles feedforward "
            "circuits so far, so it cannot reach a latch."
        )
    if intent.timing is not Timing.SUSTAINED:
        raise NotSupportedYet(
            f"timing={intent.timing.name} needs a feedback loop, and the circuit builder only assembles "
            "feedforward circuits so far."
        )
    if intent.logic is not Logic.PRESENT:
        raise NotSupportedYet(
            "combining two signals needs both of them dosed in the same assay, and the simulator varies "
            "only one signal at a time so far."
        )
    if intent.responds is Responds.PEAK:
        raise NotSupportedYet(
            "a peaked response needs two paths of opposite sign from the signal, which the circuit builder "
            "does not yet assemble."
        )


# The behaviors a library is asked about, each with the measurement it rests on.
_ASKABLE: tuple[tuple[str, Measurement], ...] = (
    ("responds UP / DOWN", Measurement.REGULATOR_STRENGTH),
    ("responds PEAK", Measurement.REGULATOR_STRENGTH),
    ("logic BOTH / EITHER / NEITHER", Measurement.TWO_INPUT_GATE),
    ("memory", Measurement.COOPERATIVITY),
    ("timing TRANSIENT", Measurement.DEGRADATION_RATE),
    ("timing OSCILLATING", Measurement.DEGRADATION_RATE),
    ("thresholds in concentration units", Measurement.DOSE_RESPONSE),
)


def capabilities_of(library: Parts) -> Capabilities:
    """What ``library``'s characterization supports, with the evidence, and what each gap would need."""
    available: list[tuple[str, str]] = []
    unavailable: list[tuple[str, Measurement]] = []
    for label, need in _ASKABLE:
        if library.has(need):
            available.append((label, _evidence(library, need)))
        else:
            unavailable.append((label, need))
    return Capabilities(tuple(available), tuple(unavailable))


def _evidence(library: Parts, need: Measurement) -> str:
    """The parts that satisfy a measurement, counted rather than asserted."""
    if need is Measurement.REGULATOR_STRENGTH:
        reps = sum(1 for c in library if c.role == "repressor")
        acts = sum(1 for c in library if c.role == "activator")
        parts = ", ".join(f"{n} {label}" for n, label in ((reps, "repressors"), (acts, "activators")) if n)
        return f"{parts}, levels measured"
    if need is Measurement.COOPERATIVITY:
        values = [c.n for c in library if c.n is not None]
        return f"Hill n measured for {len(values)} parts ({min(values):.1f}-{max(values):.1f})"
    if need is Measurement.DOSE_RESPONSE:
        n = sum(1 for c in library if c.role == "sensor" and c.K is not None and c.n is not None)
        return f"{n} sensors with a dose-response"
    if need is Measurement.ACTIVATOR:
        return f"{sum(1 for c in library if c.role == 'activator')} activators"
    return "measured"
