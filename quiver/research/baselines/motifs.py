"""Structural generators for dynamic device classes: a mutual-repression toggle (switch) and a
repressilator (oscillator). These give the search real circuits of the right family to explore, the way
`random_sensor` does for the sensor class.

Modest structural variety per draw (which regulator the reporter reads) yields a handful of distinct
topologies per family; parameters (K, n) are randomized for further variation within a topology.
"""

from __future__ import annotations

import random

from quiver.grn import GRN, Edge, Node, ParamValue

_REPORTER: dict[str, ParamValue] = {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0}


def _regulator(name: str, init: float = 0.0) -> Node:
    return Node(name, "regulator", name, {"degradation_rate": 1.0, "init_concentration": init})


def mutual_repression_toggle(rng: random.Random) -> GRN:
    """A genetic toggle: two regulators repress each other, an inducer biases one arm, and the reporter
    reads one regulator. Sweeping the inducer flips the toggle, giving a sharp (switch-like) dose-response;
    the two-state capacity shows up under :func:`quiver.objective.dynamics.hysteresis`.
    """
    k = round(rng.uniform(0.5, 2.0), 3)
    n = rng.choice([2.0, 3.0, 4.0])
    readout = rng.choice(["regA", "regB"])  # which arm the reporter reports, a structural choice
    activation = [0.0, 100.0]
    repression = [100.0, 0.0]
    nodes = [
        Node("sup", "supplement", "aTc", {}),
        _regulator("regA"),
        _regulator("regB"),
        Node("ind", "receiver", "ind", {"alpha": activation, "K": k, "n": n}),  # aTc drives regA
        Node("repA", "hill1", "repA", {"alpha": repression, "K": k, "n": n}),  # regB represses regA
        Node("repB", "hill1", "repB", {"alpha": repression, "K": k, "n": n}),  # regA represses regB
        Node("out", "reporter", "GFP", dict(_REPORTER)),
        Node("rep", "hill1", "rep", {"alpha": activation, "K": k, "n": n}),  # readout drives GFP
    ]
    edges = [
        Edge("sup", "ind"),
        Edge("ind", "regA"),
        Edge("regB", "repA"),
        Edge("repA", "regA"),
        Edge("regA", "repB"),
        Edge("repB", "regB"),
        Edge(readout, "rep"),
        Edge("rep", "out"),
    ]
    return GRN(nodes=tuple(nodes), edges=tuple(edges))


def repressilator(rng: random.Random) -> GRN:
    """A three-gene repression ring (A ⊣ B ⊣ C ⊣ A) with one regulator started high to break symmetry, and
    the reporter reading one arm. The canonical oscillator motif. The regulators and reporter carry nonzero
    degradation so the readout *tracks* the ring instead of integrating; simulate under a long constant-
    growth window (`long_run`) so a limit cycle can establish. Parameters are seeded in a regime known to
    sustain oscillation (cooperative repression, comparable degradation and dilution) and tuned from there.
    """
    k = round(rng.uniform(0.5, 2.0), 3)
    n = rng.choice([2.0, 3.0, 4.0])
    deg = round(rng.uniform(0.4, 0.8), 3)
    readout = rng.choice(["regA", "regB", "regC"])
    repression = [100.0, 0.0]
    activation = [0.0, 100.0]

    def reg(name: str, init: float) -> Node:
        return Node(name, "regulator", name, {"degradation_rate": deg, "init_concentration": init})

    nodes = [
        reg("regA", 10.0),
        reg("regB", 0.0),
        reg("regC", 0.0),
        Node("rAB", "hill1", "rAB", {"alpha": repression, "K": k, "n": n}),
        Node("rBC", "hill1", "rBC", {"alpha": repression, "K": k, "n": n}),
        Node("rCA", "hill1", "rCA", {"alpha": repression, "K": k, "n": n}),
        Node("out", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": deg, "init_concentration": 0.0}),
        Node("rep", "hill1", "rep", {"alpha": activation, "K": k, "n": n}),
    ]
    edges = [
        Edge("regA", "rAB"),
        Edge("rAB", "regB"),
        Edge("regB", "rBC"),
        Edge("rBC", "regC"),
        Edge("regC", "rCA"),
        Edge("rCA", "regA"),
        Edge(readout, "rep"),
        Edge("rep", "out"),
    ]
    return GRN(nodes=tuple(nodes), edges=tuple(edges))
