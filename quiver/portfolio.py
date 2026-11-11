"""The design call: say what you want the cell to do, get back a quiver of candidates.

:func:`design` takes the behavior as four orthogonal choices (:mod:`quiver.intent`) and parts looked up in
your own library (:mod:`quiver.parts`). No argument describes the behavior as a number --- a quantity is
something this library measures and reports, never something a design request has to supply. The parts
carry the library they came from, so a design is grounded by construction and a circuit cannot mix two
measurement conditions.

A request is checked three ways before any search runs: for choices that contradict each other, for a
behavior the parts cannot be judged against, and for one this library cannot yet construct. The last two
are separate failures, because one is a gap in the data and the other a gap in the tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from huggingface_hub import ModelHubMixin

from quiver.component import ComponentLibrary
from quiver.grn import GRN
from quiver.intent import (
    Intent,
    Logic,
    Prefer,
    Reads,
    Responds,
    Timing,
    check_characterized,
    check_consistent,
    check_supported,
)
from quiver.learn import amortized, gnn
from quiver.learn.amortized import Brief
from quiver.objective.simulate import SimConfig, simulate
from quiver.parts import DifferentConditions, Inducer, Organism, Parts, Provenance, Reporter, Strain
from quiver.parts import parts as parts_from_csv
from quiver.report import Report, acceptance, buildable, margins, measure


@dataclass(frozen=True)
class Portfolio:
    """A batch of candidate circuits, each with what it does.

    A simulation is only a proxy for behavior in a cell, so the batch is the deliverable: candidates that
    fail for different reasons are likelier to contain one that works than the single best-scoring design.
    Iterates as ``(circuit, report)`` pairs.
    """

    entries: tuple[tuple[GRN, Report], ...]
    asked: str = ""
    # The nearest candidate that fell short, kept only to explain an empty batch.
    closest: tuple[GRN, Report] | None = None

    def __iter__(self) -> Iterator[tuple[GRN, Report]]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[GRN, Report]:
        return self.entries[index]

    def __repr__(self) -> str:
        if not self.entries:
            if self.closest is None:
                return "no candidates"
            unmet = self.closest[1].unmet()
            why = unmet[0].shortfall() if unmet else "unknown"
            return f"nothing did what you asked. the closest fell short on {why}"
        layouts = len({grn.topology_hash() for grn, _ in self.entries})
        return f"{len(self.entries)} candidates that do what you asked, " f"{layouts} different circuit layouts"

    @property
    def best(self) -> tuple[GRN, Report]:
        """The highest-scoring candidate."""
        return self.entries[0]

    @property
    def circuits(self) -> list[GRN]:
        """The candidate circuits, best first."""
        return [grn for grn, _ in self.entries]

    def networks(self) -> list[object]:
        """Every candidate compiled to a LOICA ``GeneticNetwork``."""
        return [grn.to_loica() for grn, _ in self.entries]

    def sorted_by(self, prefer: Prefer) -> "Portfolio":
        """The same candidates, re-ranked by a preference, without designing again."""
        ranked = sorted(self.entries, key=lambda entry: entry[1].rank_by(prefer), reverse=True)
        return Portfolio(tuple(ranked), self.asked, self.closest)

    def best_for(self, prefer: Prefer) -> tuple[GRN, Report]:
        """The candidate that best satisfies a preference."""
        return self.sorted_by(prefer).entries[0]

    def build_sheet(self) -> str:
        """The batch as something to hand to the bench: each candidate and the parts it is built from."""
        lines = [self.asked, ""] if self.asked else []
        for index, (grn, _) in enumerate(self.entries, start=1):
            live = grn.effective()
            names = [n.name for n in grn.nodes if n.component_id is not None and n.id in live]
            lines.append(f"{index}. {' + '.join(names)}")
        return "\n".join(lines)


class Designer(ModelHubMixin):  # type: ignore[misc]
    """A trained sampler: train once, then design any request it was trained across, with no retraining.

    Amortization is what a GFlowNet offers over a one-shot search, so it is the core of this API rather
    than a variant of it: :func:`design` is the one-request case of the same machinery. Because the policy
    is conditioned on each sensor's *measured properties* rather than its name, a trained designer also
    handles an inducer it never saw.

    A trained designer follows the Hugging Face convention: :meth:`save_pretrained` writes it to a
    directory, :meth:`from_pretrained` loads it back, and :meth:`push_to_hub` publishes it. The parts it
    was trained on are saved alongside the weights, because a designer without its library cannot ground
    anything.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        library: Parts,
        *,
        conditioning: int,
        reads: Reads = Reads.SUMMARY,
        max_steps: int = 8,
    ) -> None:
        self.policy = policy
        self.library = library
        self.conditioning = conditioning
        self.reads = reads
        self.max_steps = max_steps

    def __repr__(self) -> str:
        return (
            f"Designer(trained on {self.library.name!r}, measured in {self.library.strain}, "
            f"reads={self.reads.name.lower()})"
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the policy weights, its configuration, and the parts it was trained on."""
        torch.save(self.policy.state_dict(), save_directory / "policy.pt")
        self.library.to_csv(save_directory / "parts.csv")
        config = {
            "format": "quiver-designer",
            "conditioning": self.conditioning,
            "reads": self.reads.name,
            "max_steps": self.max_steps,
            "library": self.library.name,
            "strain": self.library.strain.name,
            "organism": self.library.strain.organism.name,
            "provenance": self.library.provenance.name,
        }
        (save_directory / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        **kwargs: object,
    ) -> "Designer":
        """Rebuild a designer, and the library it was trained on, from a resolved local directory."""
        directory = Path(model_id)
        config = json.loads((directory / "config.json").read_text())
        library = parts_from_csv(
            directory / "parts.csv",
            strain=Strain(config["strain"], organism=Organism[config["organism"]]),
            provenance=Provenance[config["provenance"]],
            name=config["library"],
        )
        reads = Reads[config.get("reads", Reads.SUMMARY.name)]
        gates = len(library.as_library().gates())
        if reads is Reads.GRAPH:
            policy = gnn.conditioned_policy(config["conditioning"], n_components=gates)
        else:
            policy = amortized.conditioned_policy(config["conditioning"], n_components=gates)
        policy.load_state_dict(torch.load(directory / "policy.pt", map_location="cpu"))
        policy.eval()
        return cls(
            policy,
            library,
            conditioning=config["conditioning"],
            reads=reads,
            max_steps=config["max_steps"],
        )

    def design(
        self,
        *,
        reports: Reporter,
        senses: Inducer | tuple[Inducer, ...] = (),
        responds: Responds = Responds.UP,
        logic: Logic = Logic.PRESENT,
        memory: bool = False,
        timing: Timing = Timing.SUSTAINED,
        prefer: Prefer | None = None,
        candidates: int = 8,
        explore: int = 120,
        seed: int = 0,
    ) -> Portfolio:
        """Design a batch for one request by sampling the trained policy. No retraining."""
        intent = _resolve(reports, senses, responds, logic, memory, timing)
        if intent.library.conditions != self.library.conditions:
            raise DifferentConditions(
                f"this designer was trained on parts measured in {self.library.strain}, but the request "
                f"uses parts measured in {intent.library.strain}."
            )
        if len(intent.features()) != self.conditioning:
            raise ValueError(
                f"this designer was trained on {self.conditioning}-number requests but this one encodes to "
                f"{len(intent.features())}; it was trained by a different version."
            )
        library = self.library.as_library()
        sampler = gnn if self.reads is Reads.GRAPH else amortized
        sampled = sampler.sample_conditioned(
            self.policy,
            _brief(intent),
            explore,
            max_steps=self.max_steps,
            seed=seed,
            library=library,
        )
        return _collect(sampled, intent, library, prefer=prefer, candidates=candidates)


def train(
    library: Parts,
    *,
    reports: Reporter,
    over: Sequence[Inducer] | None = None,
    responds: Sequence[Responds] = (Responds.UP, Responds.DOWN),
    reads: Reads = Reads.SUMMARY,
    steps: int = 60,
    batch: int = 8,
    beta: float = 4.0,
    max_steps: int = 8,
    seed: int = 0,
) -> Designer:
    """Train one policy across a range of requests, so each can then be designed cheaply.

    ``over`` is the signals to amortize across, defaulting to every molecule ``library`` can sense, and
    ``responds`` the directions. The trained :class:`Designer` designs any combination of them --- and,
    because conditioning is on measured sensor properties rather than identity, signals it did not train
    on. ``reads`` selects how the policy sees a part-built circuit, which is a representation choice within
    one strategy rather than a different strategy.
    """
    signals = tuple(over) if over is not None else tuple(library.inducer(m) for m in library.detectable())
    if not signals:
        raise ValueError(f"{library.name or 'this library'} has no sensors to design around")
    briefs: list[Brief] = []
    for inducer in signals:
        for direction in responds:
            intent = _resolve(reports, inducer, direction, Logic.PRESENT, False, Timing.SUSTAINED)
            _validate(intent)
            briefs.append(_brief(intent))
    sampler = gnn if reads is Reads.GRAPH else amortized
    policy = sampler.train_conditioned(
        briefs,
        steps=steps,
        batch=batch,
        beta=beta,
        max_steps=max_steps,
        seed=seed,
        library=library.as_library(),
    )
    return Designer(policy, library, conditioning=len(briefs[0].encoding), reads=reads, max_steps=max_steps)


def design(
    *,
    reports: Reporter,
    senses: Inducer | tuple[Inducer, ...] = (),
    responds: Responds = Responds.UP,
    logic: Logic = Logic.PRESENT,
    memory: bool = False,
    timing: Timing = Timing.SUSTAINED,
    prefer: Prefer | None = None,
    reads: Reads = Reads.SUMMARY,
    candidates: int = 8,
    explore: int = 120,
    steps: int = 40,
    batch: int = 8,
    seed: int = 0,
) -> Portfolio:
    """Design a batch of circuits that do what you describe.

    ``reports`` is the readout and ``senses`` the signal or signals, both looked up in your own library.
    ``responds``, ``logic``, ``memory`` and ``timing`` are the behavior. ``prefer`` favors a direction among
    candidates that otherwise do the job, and can also be applied afterwards with
    :meth:`Portfolio.sorted_by`. ``candidates`` is how many designs come back, and ``explore`` how many are
    sampled to choose them from.

    This trains a policy for this one request and discards it. To design many requests, train a
    :class:`Designer` once with :func:`train` and reuse it.
    """
    intent = _resolve(reports, senses, responds, logic, memory, timing)
    _validate(intent)
    sampler = gnn if reads is Reads.GRAPH else amortized
    policy = sampler.train_conditioned(
        [_brief(intent)],
        steps=steps,
        batch=batch,
        beta=4.0,
        seed=seed,
        library=intent.library.as_library(),
    )
    designer = Designer(policy, intent.library, conditioning=len(intent.features()), reads=reads)
    return designer.design(
        reports=reports,
        senses=senses,
        responds=responds,
        logic=logic,
        memory=memory,
        timing=timing,
        prefer=prefer,
        candidates=candidates,
        explore=explore,
        seed=seed,
    )


def _resolve(
    reports: Reporter,
    senses: Inducer | tuple[Inducer, ...],
    responds: Responds,
    logic: Logic,
    memory: bool,
    timing: Timing,
) -> Intent:
    """Normalize a design request into an :class:`~quiver.intent.Intent`."""
    inducers = (senses,) if isinstance(senses, Inducer) else tuple(senses)
    return Intent(reports=reports, senses=inducers, responds=responds, logic=logic, memory=memory, timing=timing)


def _validate(intent: Intent) -> None:
    """Refuse a request that contradicts itself, outruns the parts, or outruns the circuit builder."""
    _check_one_library(intent)
    check_consistent(intent)
    check_characterized(intent)
    check_supported(intent)


def _brief(intent: Intent) -> Brief:
    """The conditioning point a sampler trains on or samples for."""
    return Brief(
        encoding=tuple(intent.features()),
        inducer=intent.senses[0].molecule if intent.senses else "",
        behavior=acceptance(intent),
    )


def _check_one_library(intent: Intent) -> None:
    """Refuse a design whose parts were measured under different conditions."""
    libraries = [inducer.library for inducer in intent.senses]
    if intent.reports.library is not None:
        libraries.append(intent.reports.library)
    for other in libraries[1:]:
        if other.conditions != libraries[0].conditions:
            raise DifferentConditions(
                f"these parts come from libraries measured in {libraries[0].strain} and {other.strain}. "
                "One circuit cannot mix measurement conditions."
            )


def _collect(
    sampled: list[GRN],
    intent: Intent,
    library: ComponentLibrary,
    *,
    prefer: Prefer | None,
    candidates: int,
) -> Portfolio:
    """Score the sampled circuits and return the ones that do what was asked.

    A candidate that already fails in simulation, or that leaves a stage with no part to order, is the
    cheapest possible rejection, so it is not delivered: the batch spans designs that work and might still
    fail differently at the bench. When nothing works, the nearest miss is kept so the batch can say which
    requirement fell short.
    """
    config = SimConfig()
    asked = intent.describe()
    best: dict[str, tuple[GRN, Report]] = {}
    for candidate in sampled:
        # A structural move can leave a branch that no longer reaches the reporter. Pruning it changes no
        # behavior and makes two designs that differ only in dead weight one design.
        grn = candidate.pruned()
        result = simulate(grn, config)
        measured = measure(grn, result, intent, library)
        requirements = margins(measured, result, intent)
        if library is not None:
            requirements += (buildable(grn),)
        report = Report(asked=asked, measured=measured, requirements=requirements)
        # Keyed on wiring *and* parts: two circuits wired alike but built from different transcription
        # factors are different constructs that fail for different reasons at the bench, which is the
        # diversity worth delivering.
        key = grn.structural_hash()
        if key not in best or report.score > best[key][1].score:
            best[key] = (grn, report)
    ranked = sorted(best.values(), key=lambda entry: entry[1].score, reverse=True)
    working = [entry for entry in ranked if entry[1].meets]
    closest = next((entry for entry in ranked if not entry[1].meets), None)
    portfolio = Portfolio(tuple(working[:candidates]), asked, closest if not working else None)
    return portfolio.sorted_by(prefer) if prefer is not None else portfolio
