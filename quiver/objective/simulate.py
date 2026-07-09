"""The simulation objective's engine: build a GRN's LOICA network in process, run a dose-response assay,
and return the reporter dynamics as a typed result.

LOICA is imported lazily and lives behind the ``sim`` extra, so the package imports without it; only a
call to :func:`simulate` needs it installed. The GRN's operators are wired from the graph edges exactly
as :mod:`quiver.grn` defines them: an operator's inputs are the species feeding it (ordered by port) and
its outputs are the species it produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from quiver.grn import GRN, Node


@dataclass(frozen=True)
class SimConfig:
    """Dose-sweep and assay configuration for the simulation objective."""

    dose_min: float = 1e-4
    dose_max: float = 100.0
    dose_points: int = 12
    n_measurements: int = 50
    interval: float = 0.24
    # Gompertz biomass parameters [y0, ymax, um, lambda], used unless ``sustained_growth`` is set.
    biomass: tuple[float, float, float, float] = (0.05, 1.0, 1.0, 1.0)
    # Constant per-hour growth rate for autonomous dynamics: replaces the saturating Gompertz metabolism
    # with exponential growth, so dilution/drive stay constant and a circuit can settle or oscillate
    # instead of freezing once growth plateaus. ``None`` keeps the Gompertz (characterization) model.
    sustained_growth: float | None = None
    # Integrate the circuit with LOICA's Gillespie stochastic simulation instead of the deterministic ODE,
    # so the reporter dynamics carry intrinsic molecular noise. Seed numpy's global RNG before the call for
    # a reproducible realization.
    stochastic: bool = False


@dataclass(frozen=True)
class SimulationResult:
    """Reporter dynamics from an assay: the dose sweep, the measurement times, and per-signal traces.

    ``signals`` maps a reporter signal id to a matrix of shape ``[n_doses][n_times]``; sample order is
    dose order (the zero-dose baseline first).
    """

    doses: list[float]
    times: list[float]
    signals: dict[str, list[list[float]]] = field(default_factory=dict)

    def endpoints(self, signal: str) -> list[float]:
        """The final measured value of ``signal`` at each dose, the dose-response curve."""
        return [trace[-1] if trace else 0.0 for trace in self.signals.get(signal, [])]


def dose_sweep(config: SimConfig) -> list[float]:
    """The log-spaced dose points, prefixed with a zero-dose baseline."""
    lo = math.log10(max(config.dose_min, 1e-12))
    hi = math.log10(max(config.dose_max, 1e-9))
    points = max(1, config.dose_points)
    if points == 1:
        return [0.0, 10.0**hi]
    step = (hi - lo) / (points - 1)
    return [0.0] + [10.0 ** (lo + step * i) for i in range(points)]


def long_run(sustained_growth: float = 0.4, hours: float = 120.0, n_measurements: int = 240) -> SimConfig:
    """A long, constant-growth window for *autonomous* dynamics (bistable states settling to two branches,
    or a limit cycle establishing) rather than the short saturating window a dose-response sensor uses."""
    return SimConfig(n_measurements=n_measurements, interval=hours / n_measurements, sustained_growth=sustained_growth)


def _require_loica() -> Any:
    from quiver._extras import require

    return require("loica", "sim")


def _compile(lc: Any, grn: GRN) -> tuple[Any, dict[str, Any]]:
    """Build the LOICA ``GeneticNetwork`` for ``grn`` and the map from GRN node id to LOICA object.

    The object map is returned alongside the network because a dose-response assay sets the supplement on
    each sample by its LOICA handle, which is not otherwise reachable from the network."""
    objects: dict[str, Any] = {node.id: _build_species(lc, node) for node in grn.species()}
    network = lc.GeneticNetwork()
    regulators = [objects[n.id] for n in grn.species() if n.kind == "regulator"]
    reporters = [objects[n.id] for n in grn.species() if n.kind == "reporter"]
    if regulators:
        network.add_regulator(regulators)
    if reporters:
        network.add_reporter(reporters)
    operators = [_build_operator(lc, grn, node, objects) for node in grn.operators()]
    if operators:
        network.add_operator(operators)
    return network, objects


def to_loica(grn: GRN) -> Any:
    """Compile a GRN design to a LOICA ``GeneticNetwork``: the buildable circuit behind a quiver design.

    Each GRN species becomes a LOICA ``Regulator``, ``Reporter``, or ``Supplement`` and each operator a
    LOICA ``Receiver``/``Hill1``/``Hill2``/``Source``/``Sum``, wired by the GRN's edges. The result is a
    standard LOICA object: draw it, inspect its operators, or hand it to the rest of a LOICA workflow.
    Requires the ``sim`` extra (LOICA)."""
    return _compile(_require_loica(), grn)[0]


def simulate(grn: GRN, config: SimConfig | None = None) -> SimulationResult:
    """Compile ``grn`` to a LOICA genetic network, run a dose-response assay, and return the traces."""
    lc = _require_loica()
    cfg = config or SimConfig()
    y0, ymax, um, lag = cfg.biomass
    network, objects = _compile(lc, grn)

    if cfg.sustained_growth is not None:
        rate = cfg.sustained_growth
        metab = lc.SimulatedMetabolism("sim", lambda t: 1.0, lambda t: rate)
    else:
        metab = lc.SimulatedMetabolism(
            "sim",
            lambda t: lc.gompertz(t, y0, ymax, um, lag),
            lambda t: lc.gompertz_growth_rate(t, y0, ymax, um, lag),
        )
    supplements = [n for n in grn.species() if n.kind == "supplement"]
    doses = dose_sweep(cfg) if supplements else [0.0]
    samples = []
    for dose in doses:
        sample = lc.Sample(genetic_network=network, metabolism=metab, media="M9", strain="E. coli")
        if supplements:
            sample.set_supplement(objects[supplements[0].id], dose)
        samples.append(sample)

    assay = lc.Assay(samples, n_measurements=cfg.n_measurements, interval=cfg.interval, biomass_signal_id="od")
    assay.run(stochastic=cfg.stochastic)
    return _parse_measurements(assay.measurements, doses)


def _build_species(lc: Any, node: Node) -> Any:
    if node.kind == "regulator":
        return lc.Regulator(
            name=node.name,
            degradation_rate=node.params.get("degradation_rate", 0.0),
            init_concentration=node.params.get("init_concentration", 0.0),
        )
    if node.kind == "reporter":
        kwargs: dict[str, Any] = {
            "name": node.name,
            "degradation_rate": node.params.get("degradation_rate", 0.0),
            "init_concentration": node.params.get("init_concentration", 0.0),
        }
        signal = node.params.get("signal_id")
        if isinstance(signal, str) and signal:
            kwargs["signal_id"] = signal
        return lc.Reporter(**kwargs)
    return lc.Supplement(name=node.name)


def _build_operator(lc: Any, grn: GRN, node: Node, objects: dict[str, Any]) -> Any:
    inputs = [objects[i] for i in grn.inputs_of(node.id) if i in objects]
    outputs = [objects[o] for o in grn.outputs_of(node.id) if o in objects]
    cls = {"source": lc.Source, "receiver": lc.Receiver, "hill1": lc.Hill1, "hill2": lc.Hill2, "sum": lc.Sum}[node.kind]
    kwargs = dict(node.params)
    kwargs["output"] = _one_or_list(outputs)
    if node.kind != "source":
        kwargs["input"] = inputs[0] if len(inputs) == 1 else inputs
    return cls(name=node.name, **kwargs)


def _one_or_list(items: list[Any]) -> Any:
    if not items:
        return None
    return items[0] if len(items) == 1 else items


def _parse_measurements(measurements: Any, doses: list[float]) -> SimulationResult:
    frame = measurements[measurements.Signal != "Biomass"]
    sample_ids = list(dict.fromkeys(frame.Sample.tolist()))
    times = sorted({float(t) for t in frame.Time.tolist()})
    signals: dict[str, list[list[float]]] = {}
    for signal, by_signal in frame.groupby("Signal"):
        traces: list[list[float]] = []
        for sample_id in sample_ids:
            trace = by_signal[by_signal.Sample == sample_id].sort_values("Time")
            traces.append([float(v) for v in trace.Measurement.tolist()])
        signals[str(signal)] = traces
    return SimulationResult(doses=doses[: len(sample_ids)] or doses, times=times, signals=signals)
