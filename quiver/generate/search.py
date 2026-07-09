"""Classical generators over gene-regulatory networks: a known example circuit, random parameter draws,
and mutation. These are the baselines a learned sampler is measured against, and the source of the seed
corpus a landscape characterization explores.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterator, cast

from quiver.grn import GRN, Edge, Node, ParamValue, default_params


def example_inducible() -> GRN:
    """A small aTc-inducible reporter: aTc induces a receiver that drives TetR, a Hill1 inverter reads
    TetR to express GFP. Exercises every node category and produces a clear dose response under
    simulation, the known circuit smoke tests and stubs run against.
    """
    nodes = (
        Node("atc", "supplement", "aTc", default_params("supplement")),
        Node("rec", "receiver", "receiver", {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}),
        Node("tetr", "regulator", "TetR", {"degradation_rate": 1.0, "init_concentration": 0.0}),
        Node("inv", "hill1", "inverter", {"alpha": [1.0, 0.0], "K": 1.0, "n": 2.0}),
        Node("gfp", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0}),
    )
    edges = (
        Edge("atc", "rec"),
        Edge("rec", "tetr"),
        Edge("tetr", "inv"),
        Edge("inv", "gfp"),
    )
    return GRN(nodes=nodes, edges=edges)


def mutate(grn: GRN, rng: random.Random, *, jitter: float = 0.3) -> GRN:
    """A small edit: scale every numeric parameter of one node (scalars and the entries of Hill vectors
    like ``alpha`` alike) by independent log-normal multipliers. The topology is preserved; this explores
    a fixed motif's parameter neighborhood, including the expression levels that set its dynamic range."""
    with_params = [node for node in grn.nodes if node.params]
    if not with_params:
        return grn
    target = rng.choice(with_params)
    params = {key: _scale(value, rng, jitter) for key, value in target.params.items()}
    nodes = tuple(replace(node, params=params) if node.id == target.id else node for node in grn.nodes)
    return replace(grn, nodes=nodes)


def _scale(value: ParamValue, rng: random.Random, jitter: float) -> ParamValue:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value) * rng.lognormvariate(0.0, jitter)
    return cast(ParamValue, [_scale(entry, rng, jitter) for entry in value])


def proposals(base: GRN, n: int, *, seed: int = 0, jitter: float = 0.3) -> Iterator[GRN]:
    """Yield ``n`` mutated variants of ``base``, the simplest baseline proposer over a fixed motif."""
    rng = random.Random(seed)
    for _ in range(n):
        yield mutate(base, rng, jitter=jitter)
