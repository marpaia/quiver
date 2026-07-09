"""How informative the monotonicity certificate is across the sensor topology spaces.

Three quantities:

1. **Non-monotone fraction of the unconstrained random sensor space.** ``random_sensor`` draws
   feedforward circuits with two-input logic (Hill2) gates and branched reconvergence, which the
   sign-consistency certificate (``is_monotone``) can reject. This is the space the certificate
   discriminates over, and its non-monotone fraction sets how much structural filtering the certificate
   buys, reported across regulator budgets and with the logic gates switched off.
2. **Certificate soundness.** A monotone (sign-consistent) system has a monotone steady-state
   input/output map, so a certified topology's simulated dose-response is monotone. This counts false
   certifications: certified topologies whose simulated curve is not monotone.
3. **Constructive space is monotone by construction.** The GFlowNet env grows the circuit only with
   single-input stages (relay/rewire), so every node has in-degree at most one and the interaction graph
   is a tree, hence always sign-consistent. Every sampled constructive terminal is therefore certified.

Run: ``uv run python experiments/monotonicity_fraction.py`` (needs the LOICA objective for the soundness
check; the fraction and constructive checks are simulation-free).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import TypedDict

from _common import env_int

from quiver.generate.topology import diverse_topologies
from quiver.gfn.env import apply_action, available_actions, initial_sensor
from quiver.grn import GRN
from quiver.objective.behavior import monotonicity
from quiver.objective.certificate import is_monotone
from quiver.objective.simulate import simulate

RESULTS = Path(__file__).parent / "results" / "monotonicity_fraction.json"

N = env_int("MONO_N", 4000)  # random draws per (config, seed)
SEEDS = tuple(range(env_int("MONO_SEEDS", 10)))
SOUND_SAMPLES = env_int("MONO_SOUND", 500)  # certified topologies whose dose-response is checked
WALKS = env_int("MONO_WALKS", 4000)  # constructive terminals sampled
MAX_MOVES = env_int("MONO_MAX_MOVES", 8)


class _Config(TypedDict):
    max_regulators: int
    logic_prob: float
    source_prob: float


def _fraction(*, max_regulators: int, logic_prob: float, source_prob: float) -> dict[str, float]:
    """Non-monotone fraction of the random sensor space, pooled and per-seed over ``SEEDS``."""
    per_seed: list[float] = []
    total_nonmono = 0
    total = 0
    for seed in SEEDS:
        grns = diverse_topologies(
            N, seed=seed, max_regulators=max_regulators, logic_prob=logic_prob, source_prob=source_prob
        )
        nonmono = sum(1 for g in grns if not is_monotone(g))
        per_seed.append(nonmono / len(grns))
        total_nonmono += nonmono
        total += len(grns)
    return {
        "max_regulators": max_regulators,
        "logic_prob": logic_prob,
        "source_prob": source_prob,
        "n_per_seed": N,
        "seeds": len(SEEDS),
        "nonmonotone_fraction": total_nonmono / total,
        "per_seed_mean": mean(per_seed),
        "per_seed_std": pstdev(per_seed) if len(per_seed) > 1 else 0.0,
    }


def _curve_monotonicity(grn: GRN) -> float | None:
    """Monotonicity (in ``[0,1]``, 1.0 = perfectly monotone) of the simulated dose-response, or ``None``
    if the circuit produces no reporter trace."""
    result = simulate(grn)
    if not result.signals:
        return None
    signal = next(iter(result.signals))
    endpoints = result.endpoints(signal)
    if len(endpoints) < 2:
        return None
    return monotonicity(endpoints, "any")


def _soundness(*, max_regulators: int, logic_prob: float, source_prob: float) -> dict[str, float]:
    """Among certified (monotone) topologies, count those whose simulated dose-response is not monotone
    (false certifications). A sound certificate has zero."""
    checked = 0
    false_cert = 0
    worst = 1.0
    for seed in SEEDS:
        if checked >= SOUND_SAMPLES:
            break
        grns = diverse_topologies(
            N, seed=seed, max_regulators=max_regulators, logic_prob=logic_prob, source_prob=source_prob
        )
        for grn in grns:
            if checked >= SOUND_SAMPLES:
                break
            if not is_monotone(grn):
                continue
            mono = _curve_monotonicity(grn)
            if mono is None:
                continue
            checked += 1
            worst = min(worst, mono)
            if mono < 1.0 - 1e-9:
                false_cert += 1
    return {"certified_checked": checked, "false_certifications": false_cert, "worst_curve_monotonicity": worst}


def _constructive() -> dict[str, float]:
    """Every sampled constructive terminal (initial sensor grown by random relay/rewire moves) should be
    certified monotone."""
    rng = random.Random(0)
    nonmono = 0
    for _ in range(WALKS):
        grn = initial_sensor()
        for _ in range(rng.randint(0, MAX_MOVES)):
            moves = [a for a in available_actions(grn) if a != "stop"]
            if not moves:
                break
            grn = apply_action(grn, rng.choice(moves), rng)
        if not is_monotone(grn):
            nonmono += 1
    return {"walks": WALKS, "max_moves": MAX_MOVES, "nonmonotone": nonmono, "monotone_fraction": 1.0 - nonmono / WALKS}


def main() -> None:
    configs: list[_Config] = [
        {"max_regulators": 4, "logic_prob": 0.4, "source_prob": 0.5},  # learned-sampler sensor space
        {"max_regulators": 12, "logic_prob": 0.4, "source_prob": 0.5},  # decorrelation design space
        {"max_regulators": 4, "logic_prob": 0.0, "source_prob": 0.5},  # no logic gates (isolates the driver)
    ]
    random_space = [_fraction(**cfg) for cfg in configs]
    soundness = _soundness(max_regulators=4, logic_prob=0.4, source_prob=0.5)
    constructive_space = _constructive()
    result = {
        "random_space": random_space,
        "soundness": soundness,
        "constructive_space": constructive_space,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")

    print("== non-monotone fraction of the unconstrained random sensor space ==")
    for row in random_space:
        print(
            f"  max_reg={row['max_regulators']:>2} logic_prob={row['logic_prob']} "
            f"source_prob={row['source_prob']}  -> non-monotone "
            f"{row['nonmonotone_fraction']:.1%}  (per-seed {row['per_seed_mean']:.1%} +/- {row['per_seed_std']:.1%})"
        )
    s = soundness
    print(
        f"== soundness: {s['false_certifications']}/{s['certified_checked']} certified topologies simulate "
        f"NON-monotone (worst curve monotonicity {s['worst_curve_monotonicity']:.4f}) =="
    )
    c = constructive_space
    print(
        f"== constructive space: {c['nonmonotone']}/{c['walks']} non-monotone "
        f"(monotone {c['monotone_fraction']:.1%}) =="
    )
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
