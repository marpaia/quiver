"""Whether the constructive GFlowNet is injective onto topologies (a tree) or many-to-one (a DAG).

The reward-proportionality of trajectory balance with a degenerate backward policy is exact at the topology
level only if the map from action sequences to topologies is injective. If distinct sequences reach the same
topology, the object space is a DAG and the terminal marginal carries a path-multiplicity factor m(x) the
degenerate backward policy does not correct (the marginal is proportional to m(x) * reward(x)^beta, not
reward(x)^beta).

Injectivity is decidable by enumeration rather than sampling. For the sensor construction the output stage is
a single promoter with a single input, so the only topology-affecting choice a move makes is which regulator
it grafts from (relay) or points at (rewire); all other move randomness (activation, node naming) is
topology-invisible. One exhaustive walk over those choices, up to the sampler's step cap, therefore
enumerates the construction's entire support over topologies exactly.

Two constructions are reported. The **base** env (`quiver/gfn/env.py`, the method in the paper) leaves
the regulator choice to an environment kernel; enumerating that kernel's topology-affecting support is
enumerating the same (action, regulator) branching, with no cap on relay. The **structured** env
(`quiver/gfn/structured_env.py`, a variant) hands that choice to the policy and caps the addressable
regulators at MAX_REG. The structured helpers implement the canonical single-site edit both share, so they
drive both enumerations.

Run: ``uv run python experiments/injectivity.py`` (torch-free; needs no extras).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from _common import MAX_STEPS

from quiver.gfn.env import initial_sensor
from quiver.gfn.structured_env import (
    MAX_REG,
    _canonical_output_input,
    _structured_relay,
    _structured_rewire,
    regulators_in_order,
)
from quiver.grn import GRN

RESULTS = Path(__file__).parent / "results" / "injectivity.json"

# Enumeration is deterministic in topology; a fixed generator supplies the topology-invisible activation draw.
_RNG = random.Random(0)

Children = Callable[[GRN], "list[tuple[str, GRN]]"]


def _current_output_source(grn: GRN) -> str | None:
    site = _canonical_output_input(grn)
    return grn.edges[site[1]].source if site is not None else None


def base_children(grn: GRN) -> list[tuple[str, GRN]]:
    """The base env's non-stop moves, branched over every topology-affecting kernel outcome: ``relay`` from
    each existing regulator (no cap, matching ``env.available_actions``), ``rewire`` onto each other
    regulator once at least two exist. Each is a distinct construction path the kernel can realize."""
    regs = regulators_in_order(grn)
    out: list[tuple[str, GRN]] = []
    for i, r in enumerate(regs):
        child = _structured_relay(grn, r, _RNG)
        if child is not grn:
            out.append((f"relay:{i}", child))
    if len(regs) >= 2 and _canonical_output_input(grn) is not None:
        current = _current_output_source(grn)
        for i, r in enumerate(regs):
            if r == current:
                continue
            child = _structured_rewire(grn, r)
            if child is not grn:
                out.append((f"rewire:{i}", child))
    return out


def structured_children(grn: GRN) -> list[tuple[str, GRN]]:
    """The structured env's non-stop moves: the same edits as :func:`base_children`, but the addressable
    regulator index is capped at ``MAX_REG`` (relay is masked once the cap is reached)."""
    regs = regulators_in_order(grn)
    out: list[tuple[str, GRN]] = []
    if len(regs) < MAX_REG:
        for i, r in enumerate(regs):
            child = _structured_relay(grn, r, _RNG)
            if child is not grn:
                out.append((f"relay:{i}", child))
    if len(regs) >= 2 and _canonical_output_input(grn) is not None:
        current = _current_output_source(grn)
        for i, r in enumerate(regs[:MAX_REG]):
            if r == current:
                continue
            child = _structured_rewire(grn, r)
            if child is not grn:
                out.append((f"rewire:{i}", child))
    return out


def enumerate_terminals(children: Children, max_steps: int) -> list[str]:
    """The ``topology_hash`` of every terminal circuit the construction admits, one entry per distinct
    action sequence. A trajectory ends when the policy could ``stop`` (recorded here) or at ``max_steps``
    non-stop moves, mirroring :func:`quiver.learn._core.rollout`."""
    terminals: list[str] = []

    def walk(grn: GRN, depth: int) -> None:
        # ``stop`` is always available; taking it terminates the trajectory at the current circuit.
        terminals.append(grn.topology_hash())
        if depth >= max_steps:
            return
        for _, child in children(grn):
            walk(child, depth + 1)

    walk(initial_sensor(), 0)
    return terminals


def summarize(children: Children, max_steps: int) -> dict[str, object]:
    terminals = enumerate_terminals(children, max_steps)
    per_topology = Counter(terminals)
    multiplicities = Counter(per_topology.values())
    n_traj = len(terminals)
    n_topo = len(per_topology)
    collided = sum(1 for m in per_topology.values() if m > 1)
    return {
        "max_steps": max_steps,
        "n_trajectories": n_traj,
        "n_distinct_topologies": n_topo,
        "injective": collided == 0,
        "topologies_with_multiplicity_gt1": collided,
        "fraction_topologies_collided": collided / n_topo if n_topo else 0.0,
        "max_trajectories_per_topology": max(per_topology.values()) if per_topology else 0,
        "mean_trajectories_per_topology": n_traj / n_topo if n_topo else 0.0,
        "multiplicity_histogram": {str(k): v for k, v in sorted(multiplicities.items())},
    }


def main() -> None:
    result = {
        "base": summarize(base_children, MAX_STEPS),
        "structured": summarize(structured_children, MAX_STEPS),
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")

    for name, r in result.items():
        print(f"[{name}]  max_steps={r['max_steps']}")
        print(f"  terminal trajectories         {r['n_trajectories']}")
        print(f"  distinct topologies           {r['n_distinct_topologies']}")
        print(f"  injective (tree over topos)   {r['injective']}")
        print(
            f"  topologies reached >1 way     {r['topologies_with_multiplicity_gt1']}"
            f"  ({r['fraction_topologies_collided']:.1%})"
        )
        print(f"  max trajectories per topology {r['max_trajectories_per_topology']}")
        print(f"  mean trajectories per topology{r['mean_trajectories_per_topology']:.2f}")
        print(f"  multiplicity histogram        {r['multiplicity_histogram']}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
