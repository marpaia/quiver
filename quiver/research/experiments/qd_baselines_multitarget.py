"""Multi-target version of the single-target discovery comparison (review item R1).

The single-target benchmark (``qd_baselines.py``) establishes the GFlowNet's breadth on one target, where the
advantage over the strongest search baseline (evolutionary) is large but not a clean per-seed sweep. This
script asks whether that advantage generalizes across the sensor target grid: it runs every method per
target, trains a fresh single-target GFlowNet per target (the same per-target footing the search baselines
get), and reports the GFlowNet-minus-evolutionary advantage both within each target (paired over seeds) and
across targets (a sign test over the per-target mean advantages).

Run: ``uv run python experiments/qd_baselines_multitarget.py``.
Knobs: ``SEEDS`` (default 10). The run is SEEDS x len(grid) GFlowNet trainings, so ``SEEDS=5`` halves it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from _common import (
    BETA,
    EVAL_BUDGET,
    SENSOR_TARGET_GRID,
    TRAIN_BATCH,
    TRAIN_STEPS,
    TUNE_STEPS,
    Stream,
    env_int,
    portfolio,
)
from _gfn import gflownet_stream

from quiver.objective import Behavior, TargetResponse, dose_response_match
from quiver.research.baselines import evolutionary_search, gp_ucb_search, map_elites, novelty_search, random_search
from quiver.research.eval import paired_summary

SEEDS = env_int("SEEDS", 10)

RESULTS = Path(__file__).parent / "results" / "qd_baselines_multitarget.json"

METHODS: dict[str, Callable[[Behavior, int], Stream]] = {
    "random": lambda b, s: random_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "evolutionary": lambda b, s: evolutionary_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "map-elites": lambda b, s: map_elites(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "novelty": lambda b, s: novelty_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "bo": lambda b, s: gp_ucb_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "gflownet": gflownet_stream,
}


def _label(t: TargetResponse) -> str:
    return f"ec50={t.ec50:g},F={t.dynamic_range:g},{t.direction}"


def main() -> None:
    grid = SENSOR_TARGET_GRID
    print(
        f"{len(grid)} targets x {SEEDS} seeds | eval {EVAL_BUDGET} sims/method "
        f"| GFN train {TRAIN_STEPS * TRAIN_BATCH} sims/target, beta={BETA}\n"
    )
    t0 = time.perf_counter()
    per_target: list[dict[str, object]] = []
    gfn_means: list[float] = []
    evo_means: list[float] = []

    for ti, target in enumerate(grid):
        behavior = dose_response_match(target)
        distinct: dict[str, list[int]] = {m: [] for m in METHODS}
        for seed in range(SEEDS):
            for name, fn in METHODS.items():
                d, _ = portfolio(fn(behavior, seed))
                distinct[name].append(d)
        means = {name: mean(distinct[name]) for name in METHODS}
        within = paired_summary(distinct["gflownet"], distinct["evolutionary"])
        gfn_means.append(means["gflownet"])
        evo_means.append(means["evolutionary"])
        per_target.append(
            {
                "target": {"ec50": target.ec50, "dynamic_range": target.dynamic_range, "direction": target.direction},
                "distinct_mean": means,
                "distinct_std": {name: pstdev(distinct[name]) for name in METHODS},
                "gfn_minus_evolutionary": {
                    "mean_diff": within.mean_diff,
                    "ci_low": within.ci_low,
                    "ci_high": within.ci_high,
                    "win_rate": within.win_rate,
                    "sign_p": within.sign_p,
                    "n": within.n,
                },
                "per_seed_distinct": distinct,
            }
        )
        print(
            f"  [{ti + 1}/{len(grid)}] {_label(target):26s} "
            f"gfn {means['gflownet']:5.1f}  evo {means['evolutionary']:5.1f}  "
            f"delta {within.mean_diff:+.1f} (p={within.sign_p:.2g})  ({time.perf_counter() - t0:.0f}s)"
        )

    across = paired_summary(gfn_means, evo_means)
    print(
        f"\n== across {len(grid)} targets: gfn - evolutionary on per-target means ==\n"
        f"  {across.mean_diff:+.2f} [95% CI {across.ci_low:+.2f}, {across.ci_high:+.2f}] "
        f"| targets won {across.win_rate:.0%} | sign-test p={across.sign_p:.2g} | n={across.n}"
    )

    result = {
        "config": {
            "targets": len(grid),
            "seeds": SEEDS,
            "eval_budget": EVAL_BUDGET,
            "train_sims": TRAIN_STEPS * TRAIN_BATCH,
            "beta": BETA,
        },
        "across_targets_gfn_minus_evolutionary": {
            "mean_diff": across.mean_diff,
            "ci_low": across.ci_low,
            "ci_high": across.ci_high,
            "win_rate": across.win_rate,
            "sign_p": across.sign_p,
            "n": across.n,
        },
        "per_target": per_target,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
