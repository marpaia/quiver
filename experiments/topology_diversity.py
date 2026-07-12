"""Characterize the enriched (non-enumerable) sensor topology space.

The baseline search ran on a small, enumerable topology set (depth-only cascades). This re-characterizes
the space produced by the enriched generator (branched / two-input-logic / constitutive-source
feedforward circuits) to confirm the two things a learned sampler needs: the topology space is large
(non-enumerable), and it still carries a graded, multimodal objective (many distinct topologies hit the
target to varying degrees).

Run: ``uv run python experiments/topology_diversity.py``.
"""

from __future__ import annotations

import random
import time
from statistics import mean, pstdev

from quiver.eval import select_portfolio
from quiver.generate import diverse_topologies, mutate
from quiver.grn import GRN
from quiver.objective import Behavior, TargetResponse, dose_response_match, score_once

TARGET = TargetResponse(signal="GFP", ec50=1.0, dynamic_range=100.0, direction="any")
POPULATION = 80
TUNE_STEPS = 4
PASS = 0.5


def tune(grn: GRN, behavior: Behavior, rng: random.Random, steps: int) -> float:
    best = score_once(grn, behavior)
    current = grn
    for _ in range(steps):
        candidate = mutate(current, rng, jitter=0.4)
        score = score_once(candidate, behavior)
        if score > best:
            current, best = candidate, score
    return best


def main() -> None:
    behavior = dose_response_match(TARGET)
    rng = random.Random(0)
    population = diverse_topologies(POPULATION, seed=0)
    n_topologies = len({grn.topology_hash() for grn in population})
    print(f"generated {POPULATION} circuits; {n_topologies} distinct topologies")

    t0 = time.perf_counter()
    scored = [(grn.topology_hash(), tune(grn, behavior, rng, TUNE_STEPS)) for grn in population]
    elapsed = time.perf_counter() - t0
    scores = [score for _, score in scored]
    per_sim_ms = elapsed / (POPULATION * (TUNE_STEPS + 1)) * 1000
    print(f"tuned + scored {POPULATION} candidates in {elapsed:.1f}s (~{per_sim_ms:.0f} ms/sim)")
    print(f"score span: min {min(scores):.3f} | mean {mean(scores):.3f} ± {pstdev(scores):.3f} | max {max(scores):.3f}")

    passer_topologies = {topo for topo, score in scored if score >= PASS}
    n_passers = sum(1 for score in scores if score >= PASS)
    print(f"usable sensors (score >= {PASS}): {n_passers} / {POPULATION}")
    print(f"distinct topologies among passers: {len(passer_topologies)}")
    portfolio = select_portfolio([(topo, score) for topo, score in scored if score >= PASS], k=10)
    print(f"portfolio (top distinct designs): {len(portfolio)} | scores {[round(s, 2) for _, s in portfolio]}")

    print("\n== verdict ==")
    span = max(scores) - min(scores)
    print(f"  topology space large / non-enumerable: {n_topologies} distinct in {POPULATION} draws")
    print(f"  graded objective varies (span {span:.2f}): {span > 0.3}")
    print(f"  multimodal, distinct topologies pass: {len(passer_topologies)}")


if __name__ == "__main__":
    main()
