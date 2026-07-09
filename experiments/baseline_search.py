"""Discovery curves for the no-learning topology searches.

The fair metric for a design search is the discovery curve: distinct high-scoring topologies found per
simulation budget. This runs random search and evolutionary search over the sensor topology space at an
equal budget and reports their curves, the control a learned diversity-seeking sampler (a constructive
GFlowNet) must beat on portfolio breadth.

Run: ``uv run python experiments/baseline_search.py`` (needs the ``sim`` extra installed).
"""

from __future__ import annotations

import time

from quiver.eval import PortfolioReport, evaluate_portfolio
from quiver.generate import evolutionary_search, random_search
from quiver.objective import TargetResponse, dose_response_match

TARGET = TargetResponse(signal="GFP", ec50=1.0, dynamic_range=100.0, direction="any")
BUDGET = 300
TUNE_STEPS = 4
PASS = 0.5


def report(name: str, stream: list[tuple[str, float]]) -> PortfolioReport:
    scores = [score for _, score in stream]
    hashes = [topology for topology, _ in stream]
    result = evaluate_portfolio(scores, hashes, passed=[s >= PASS for s in scores])
    points = " → ".join(str(result.modes_at(f)) for f in (0.25, 0.5, 1.0))
    print(
        f"  {name:13s} distinct high-scoring topologies {result.distinct_passers:3d}"
        f" | discovery @25/50/100% {points} | passers {result.passers}/{len(stream)}"
    )
    return result


def main() -> None:
    behavior = dose_response_match(TARGET)
    print(f"budget {BUDGET} sims each | target EC50={TARGET.ec50}, dynamic range={TARGET.dynamic_range}")
    t0 = time.perf_counter()
    random_stream = random_search(behavior, budget=BUDGET, tune_steps=TUNE_STEPS, seed=0)
    evo_stream = evolutionary_search(behavior, budget=BUDGET, tune_steps=TUNE_STEPS, seed=0)
    print(f"ran both searches in {time.perf_counter() - t0:.1f}s\n")

    print(f"discovery curves (distinct topologies scoring >= {PASS}):")
    random_report = report("random", random_stream)
    evo_report = report("evolutionary", evo_stream)

    print("\n== baseline verdict ==")
    print(f"  random: {random_report.distinct_passers} | evolutionary: {evo_report.distinct_passers} distinct designs")
    print("  this is the no-learning control the GFlowNet sampler must beat on portfolio breadth")


if __name__ == "__main__":
    main()
