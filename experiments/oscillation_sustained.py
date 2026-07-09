"""How many oscillator passers are *sustained* rather than slowly damped?

The oscillation detector scores autocorrelation periodicity, which a slowly-damped oscillation also has.
This draws repressilator variants, tunes each against the oscillation detector under the long
constant-growth window, and for the passers (score >= 0.5) reports the fraction that are *sustained*: the
envelope of the settled oscillation holds its amplitude (:func:`is_sustained`) rather than decaying, so
"sustained oscillation" is substantiated, not merely "oscillation".

Run: ``uv run python experiments/oscillation_sustained.py`` (needs the ``sim`` extra).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean

from quiver.generate import repressilator, tune_best
from quiver.objective import envelope_decay, is_sustained, long_run, simulate, sustained_oscillation

POPULATION = 30
SEEDS = 3
TUNE_STEPS = 2
PASS = 0.5
AUTONOMOUS = long_run(sustained_growth=1.0)

RESULTS = Path(__file__).parent / "results" / "oscillation_sustained.json"


def main() -> None:
    detector = sustained_oscillation("GFP")
    print(f"{POPULATION} repressilator draws/seed, {SEEDS} seeds, tune {TUNE_STEPS}, PASS {PASS}\n")
    t0 = time.perf_counter()
    passers = 0
    sustained = 0
    ratios: list[float] = []
    for seed in range(SEEDS):
        rng = random.Random(seed)
        for _ in range(POPULATION):
            grn = repressilator(rng)
            tuned, score = tune_best(grn, detector, rng, TUNE_STEPS, config=AUTONOMOUS)
            if score < PASS:
                continue
            passers += 1
            result = simulate(tuned, AUTONOMOUS)
            ratios.append(envelope_decay(result, "GFP"))
            if is_sustained(result, "GFP"):
                sustained += 1

    frac = sustained / passers if passers else 0.0
    mean_ratio = mean(ratios) if ratios else 0.0
    print(f"oscillator passers: {passers}")
    print(f"sustained (envelope holds, not damped): {sustained}/{passers} = {frac:.2f}")
    print(f"mean envelope ratio over passers: {mean_ratio:.3f}")

    summary = {
        "config": {"population": POPULATION, "seeds": SEEDS, "tune_steps": TUNE_STEPS, "pass": PASS},
        "passers": passers,
        "sustained": sustained,
        "sustained_fraction": frac,
        "mean_envelope_ratio": mean_ratio,
    }
    RESULTS.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {RESULTS}")
    print(f"\nran in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
