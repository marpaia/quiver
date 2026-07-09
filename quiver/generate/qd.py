"""Quality-diversity search over sensor topologies: MAP-Elites and Novelty Search.

These are the diversity-native baselines a diversity-seeking sampler must be measured against --- unlike
random and evolutionary search, they *explicitly* spread across a behavior space, so beating (or matching)
them isolates what the learned sampler adds. Both search the same topology space with the same per-candidate
scoring as the other baselines (`quiver.generate.optimize`), differing only in the proposal and
selection mechanism, and return a ``(topology_hash, score)`` stream in evaluation order for the discovery
metric.

The behavior descriptor is the coarse structure of a circuit --- its regulator and operator counts --- the
axis a portfolio spreads over. MAP-Elites keeps the best design per descriptor cell; Novelty Search selects
for descriptor novelty rather than score.
"""

from __future__ import annotations

import random

from quiver.generate.optimize import tune_score
from quiver.generate.topology import random_sensor, structural_mutate
from quiver.grn import GRN, is_operator
from quiver.objective import Behavior

Scored = tuple[str, float]


def _descriptor(grn: GRN) -> tuple[int, int]:
    """A circuit's behavior descriptor: (regulator count, operator count) --- the coarse structural axes a
    quality-diversity archive spreads over."""
    regulators = sum(1 for node in grn.nodes if node.kind == "regulator")
    operators = sum(1 for node in grn.nodes if is_operator(node.kind))
    return (regulators, operators)


def map_elites(
    behavior: Behavior,
    *,
    budget: int,
    tune_steps: int = 2,
    seed: int = 0,
    max_regulators: int = 4,
    init: int = 12,
) -> list[Scored]:
    """MAP-Elites: maintain one elite (best-scoring design) per descriptor cell, and mutate random elites to
    fill the archive. Seeds with ``init`` random topologies, then breeds from the archive by structural
    mutation until the budget is spent --- spreading high-scoring designs across the descriptor space."""
    rng = random.Random(seed)
    per = tune_steps + 1
    stream: list[Scored] = []
    archive: dict[tuple[int, int], tuple[GRN, float]] = {}
    for i in range(max(1, budget // per)):
        if i < init or not archive:
            grn = random_sensor(rng, max_regulators=max_regulators)
        else:
            parent = rng.choice(list(archive.values()))[0]
            grn = structural_mutate(parent, rng)
        score = tune_score(grn, behavior, rng, tune_steps)
        stream.append((grn.topology_hash(), score))
        cell = _descriptor(grn)
        if cell not in archive or score > archive[cell][1]:
            archive[cell] = (grn, score)
    return stream


def _cell_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def novelty_search(
    behavior: Behavior,
    *,
    budget: int,
    tune_steps: int = 2,
    seed: int = 0,
    pop_size: int = 12,
    max_regulators: int = 4,
    k: int = 3,
) -> list[Scored]:
    """Novelty Search: breed a population selected for descriptor *novelty* (mean distance to the ``k``
    nearest previously-seen descriptors) rather than score, so it pushes into unexplored structure. Scores
    are still recorded (for the discovery metric) but do not drive selection."""
    rng = random.Random(seed)
    per = tune_steps + 1
    stream: list[Scored] = []
    seen: list[tuple[int, int]] = []

    def novelty(descriptor: tuple[int, int]) -> float:
        if not seen:
            return 1.0
        distances = sorted(_cell_distance(descriptor, other) for other in seen)
        return sum(distances[:k]) / min(k, len(distances))

    def evaluate(grn: GRN) -> tuple[float, float]:
        score = tune_score(grn, behavior, rng, tune_steps)
        stream.append((grn.topology_hash(), score))
        descriptor = _descriptor(grn)
        nov = novelty(descriptor)
        seen.append(descriptor)
        return score, nov

    population: list[tuple[GRN, float]] = []
    for _ in range(pop_size):
        if len(stream) * per >= budget:
            return stream
        grn = random_sensor(rng, max_regulators=max_regulators)
        population.append((grn, evaluate(grn)[1]))

    while len(stream) * per < budget:
        population.sort(key=lambda pair: pair[1], reverse=True)
        survivors = population[: max(1, pop_size // 2)]
        children: list[tuple[GRN, float]] = []
        for parent, _ in survivors:
            if len(stream) * per >= budget:
                break
            child = structural_mutate(parent, rng)
            children.append((child, evaluate(child)[1]))
        population = survivors + children
    return stream
