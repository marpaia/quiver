"""Isolate the diversity contribution against diversity-native baselines.

Random and evolutionary search are not diversity-seeking, so a diversity-trained sampler beating them on a
diversity metric is close to structural. This compares the GFlowNet on the single-target discovery benchmark
against the informative competitors: MAP-Elites and Novelty Search (quality-diversity methods that
explicitly spread across a behavior space) and GP-UCB Bayesian optimization (the expensive-oracle
comparator). All search the same topology space at the same eval budget and scoring; the reported quantity
is distinct high-scoring topologies discovered.

The differentiator the GFlowNet keeps even against a strong quality-diversity baseline is amortization: the
QD and BO methods are run per target, while the GFlowNet is trained once and serves a distribution of
targets. This experiment isolates the per-target diversity comparison.

Run: ``uv run python experiments/qd_baselines.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from _common import BETA, EVAL_BUDGET, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS, Stream, portfolio
from _gfn import gflownet_stream

from quiver.eval import paired_summary
from quiver.generate import evolutionary_search, gp_ucb_search, map_elites, novelty_search, random_search
from quiver.objective import Behavior, TargetResponse, dose_response_match

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
SEEDS = 10

RESULTS = Path(__file__).parent / "results" / "qd_baselines.json"

METHODS: dict[str, Callable[[Behavior, int], Stream]] = {
    "random": lambda b, s: random_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "evolutionary": lambda b, s: evolutionary_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "map-elites": lambda b, s: map_elites(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "novelty": lambda b, s: novelty_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "bo": lambda b, s: gp_ucb_search(b, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=s),
    "gflownet": gflownet_stream,
}


def main() -> None:
    behavior = dose_response_match(TARGET)
    print(
        f"{SEEDS} seeds | eval {EVAL_BUDGET} sims/method (tune {TUNE_STEPS}) "
        f"| GFN train {TRAIN_STEPS * TRAIN_BATCH} sims, β={BETA}\n"
    )
    final: dict[str, list[int]] = {m: [] for m in METHODS}
    midway: dict[str, list[int]] = {m: [] for m in METHODS}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for name, fn in METHODS.items():
            distinct, mid = portfolio(fn(behavior, seed))
            final[name].append(distinct)
            midway[name].append(mid)
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    print("\ndistinct high-scoring topologies (mean ± std over seeds) | discovery @50% budget:")
    methods: dict[str, dict[str, float]] = {}
    for name in METHODS:
        f_m, f_s = mean(final[name]), pstdev(final[name])
        m_m = mean(midway[name])
        print(f"  {name:13s} {f_m:5.1f} ± {f_s:4.1f}   |   @50% {m_m:.1f}")
        methods[name] = {"distinct_mean": f_m, "distinct_std": f_s, "midway_mean": m_m}

    print("\n== paired advantage of gflownet over each (per seed) ==")
    paired: dict[str, dict[str, float]] = {}
    for opponent in ("map-elites", "novelty", "bo", "evolutionary", "random"):
        s = paired_summary(final["gflownet"], final[opponent])
        print(
            f"  gfn − {opponent:13s} {s.mean_diff:+.2f} [95% CI {s.ci_low:+.2f}, {s.ci_high:+.2f}] "
            f"| win {s.win_rate:.0%} | sign-test p={s.sign_p:.2g} | n={s.n}"
        )
        paired[opponent] = {
            "mean_diff": s.mean_diff,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "win_rate": s.win_rate,
            "sign_p": s.sign_p,
            "n": s.n,
        }

    result = {
        "config": {
            "seeds": SEEDS,
            "eval_budget": EVAL_BUDGET,
            "tune_steps": TUNE_STEPS,
            "train_sims": TRAIN_STEPS * TRAIN_BATCH,
            "beta": BETA,
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "methods": methods,
        "paired_gflownet_minus": paired,
        "per_seed_distinct": final,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
