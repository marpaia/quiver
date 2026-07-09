"""Breadth: the design pipeline across device classes, not just the sensor.

For each device class (sensor, switch, oscillator) it draws a population of that family's topologies,
tunes each against the class's behavior detector, and reports distinct high-scoring designs, the same
discovery/portfolio metric used elsewhere, now shown to work per class. It also demonstrates the
two-simulation hysteresis check (bistability capacity) on the best toggle vs. a sensor.

Run: ``uv run python experiments/device_classes.py`` (needs the ``sim`` extra installed).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from quiver.eval import evaluate_portfolio
from quiver.generate import mutual_repression_toggle, random_sensor, repressilator, tune_score
from quiver.grn import GRN
from quiver.objective import (
    Behavior,
    SimConfig,
    hysteresis,
    inducible_response,
    long_run,
    sustained_oscillation,
    ultrasensitive_switch,
)

POPULATION = 30
TUNE_STEPS = 2
PASS = 0.5
SEEDS = 3
AUTONOMOUS = long_run(sustained_growth=1.0)  # long constant-growth window for oscillation / settling

RESULTS = Path(__file__).parent / "results" / "device_classes.json"

# Each class: a generator, its behavior detector, and the simulation mode it needs (None = dose-response).
CLASSES: list[tuple[str, Callable[[random.Random], GRN], Behavior, SimConfig | None]] = [
    ("sensor", lambda rng: random_sensor(rng), inducible_response("GFP"), None),
    ("switch", mutual_repression_toggle, ultrasensitive_switch("GFP"), None),
    ("oscillator", repressilator, sustained_oscillation("GFP"), AUTONOMOUS),
]


def run_class(
    generator: Callable[[random.Random], GRN], detector: Behavior, seed: int, config: SimConfig | None
) -> tuple[int, int]:
    rng = random.Random(seed)
    scored: list[tuple[str, float]] = []
    for _ in range(POPULATION):
        grn = generator(rng)
        scored.append((grn.topology_hash(), tune_score(grn, detector, rng, TUNE_STEPS, config=config)))
    scores = [s for _, s in scored]
    report = evaluate_portfolio(scores, [t for t, _ in scored], passed=[s >= PASS for s in scores])
    return report.distinct_passers, report.passers


def main() -> None:
    print(f"{POPULATION} draws/class, {SEEDS} seeds, tune {TUNE_STEPS}, PASS {PASS}\n")
    print(f"{'device class':<12} | {'distinct high-scoring designs':>30} | {'passers/pop':>12}")
    t0 = time.perf_counter()
    rows: list[dict[str, str | float]] = []
    for name, generator, detector, config in CLASSES:
        distinct = [run_class(generator, detector, s, config) for s in range(SEEDS)]
        d_mean, d_std = mean(d for d, _ in distinct), pstdev([d for d, _ in distinct])
        p_mean = mean(p for _, p in distinct)
        print(f"{name:<12} | {f'{d_mean:.1f} ± {d_std:.1f}':>30} | {f'{p_mean:.1f}/{POPULATION}':>12}")
        rows.append(
            {
                "device_class": name,
                "distinct_mean": d_mean,
                "distinct_std": d_std,
                "passers_mean": p_mean,
                "population": POPULATION,
            }
        )

    # Bistability capacity, under the long autonomous window so states settle to their stable branches:
    # the toggle shows history-dependence (hysteresis); a sensor does not.
    toggle = max((mutual_repression_toggle(random.Random(s)) for s in range(8)), key=lambda g: len(g.nodes))
    sensor = random_sensor(random.Random(0))
    toggle_h = hysteresis(toggle, "GFP", config=AUTONOMOUS)
    sensor_h = hysteresis(sensor, "GFP", config=AUTONOMOUS)
    print(f"\nhysteresis (bistability capacity): toggle {toggle_h:.3f} | sensor {sensor_h:.3f}")

    result = {
        "config": {"population": POPULATION, "seeds": SEEDS, "tune_steps": TUNE_STEPS, "pass": PASS},
        "classes": rows,
        "hysteresis": {"toggle": toggle_h, "sensor": sensor_h},
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS}")
    print(f"\nran in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
