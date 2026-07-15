"""W1: does a policy that reads circuit physics beat one that reads a feature summary?

A fair, head-to-head comparison of the two structured-env policies under the buildability-aware objective:
the feature-summary MLP (:mod:`quiver.learn.structured`) versus the graph neural network over the circuit
graph (:mod:`quiver.learn.gnn`). Both search the identical action space at the identical budget and seeds
and train against :func:`quiver.objective.buildable_objective`; only the policy representation differs, so a
difference is attributable to the architecture.

The reported quantities span the decision axes: **breadth** (distinct high-scoring topologies), **quality**
(mean and peak score of the delivered portfolio), and **buildability** (the portfolio's mean metabolic
burden, cascade depth, and realizable-parameter fraction). The graph policy is adopted only if it matches or
beats the MLP on breadth at equal quality *and* delivers more buildable portfolios; a tie is a reportable
outcome (small circuit graphs may not need a graph policy).

Run: ``uv run python experiments/gnn_benchmark.py``. Scale with ``SEEDS=`` and ``BETAS=`` (comma-separated).
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402

from _common import EVAL_BUDGET, MAX_STEPS, PASS, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS, env_int  # noqa: E402
from _policy import deliver, summarize  # noqa: E402

from quiver.learn import gnn, structured  # noqa: E402
from quiver.objective import TargetResponse, buildable_objective, dose_response_match  # noqa: E402
from quiver.research.eval import paired_summary  # noqa: E402

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")

SEEDS = env_int("SEEDS", 3)
BETAS = [float(b) for b in os.environ.get("BETAS", "1,4").split(",") if b.strip()]
N_SAMPLES = EVAL_BUDGET // (TUNE_STEPS + 1)
ARMS = {"mlp": structured, "gnn": gnn}

RESULTS = Path(__file__).parent / "results" / "gnn_benchmark.json"

# Buildability metrics reported per arm, alongside breadth (distinct) and quality (peak/mean score).
BUILD_KEYS = ("burden_mean", "regulators_mean", "realizable_frac")


def main() -> None:
    behavior = dose_response_match(TARGET)
    objective = buildable_objective(behavior)
    print(
        f"{SEEDS} seeds | β ∈ {BETAS} | deliver {N_SAMPLES} samples (tune {TUNE_STEPS}) "
        f"| train {TRAIN_STEPS * TRAIN_BATCH} evals under buildable_objective (ensemble 8)\n"
    )

    # per_seed[arm][beta] -> list of per-seed summary dicts.
    per_seed: dict[str, dict[float, list[dict[str, float]]]] = {a: {b: [] for b in BETAS} for a in ARMS}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for beta in BETAS:
            for arm, module in ARMS.items():
                designs = deliver(
                    module,
                    behavior,
                    objective=objective,
                    beta=beta,
                    seed=seed,
                    n_samples=N_SAMPLES,
                    tune_steps=TUNE_STEPS,
                    train_steps=TRAIN_STEPS,
                    train_batch=TRAIN_BATCH,
                    max_steps=MAX_STEPS,
                )
                per_seed[arm][beta].append(summarize(designs, pass_threshold=PASS))
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    arms: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARMS:
        arms[arm] = {}
        for beta in BETAS:
            rows = per_seed[arm][beta]
            distinct = [r["distinct"] for r in rows]
            arms[arm][f"beta={beta}"] = {
                "distinct_mean": mean(distinct),
                "distinct_std": pstdev(distinct),
                "peak_mean": mean(r["peak_score"] for r in rows),
                "meanscore_mean": mean(r["mean_score"] for r in rows),
                **{k: mean(r[k] for r in rows) for k in BUILD_KEYS},
                "per_seed_distinct": distinct,
            }

    print("\ndistinct high-scoring | peak | mean | burden | regs | realizable, per arm × β:")
    for beta in BETAS:
        print(f"  β = {beta}")
        for arm in ARMS:
            m = arms[arm][f"beta={beta}"]
            print(
                f"    {arm:4s} {m['distinct_mean']:5.1f} ± {m['distinct_std']:4.1f} | "
                f"peak {m['peak_mean']:.3f} | mean {m['meanscore_mean']:.3f} | "
                f"burden {m['burden_mean']:.3f} | regs {m['regulators_mean']:.2f} | "
                f"real {m['realizable_frac']:.2f}"
            )

    print("\n== paired advantage of gnn over mlp (distinct high-scoring, per seed) ==")
    paired: dict[str, dict[str, float]] = {}
    for beta in BETAS:
        gnn_d = [r["distinct"] for r in per_seed["gnn"][beta]]
        mlp_d = [r["distinct"] for r in per_seed["mlp"][beta]]
        s = paired_summary(gnn_d, mlp_d)
        print(
            f"  β={beta}: {s.mean_diff:+.2f} [95% CI {s.ci_low:+.2f}, {s.ci_high:+.2f}] "
            f"| win {s.win_rate:.0%} | sign-test p={s.sign_p:.2g} | n={s.n}"
        )
        paired[f"beta={beta}"] = {
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
            "betas": BETAS,
            "deliver_samples": N_SAMPLES,
            "tune_steps": TUNE_STEPS,
            "train_evals": TRAIN_STEPS * TRAIN_BATCH,
            "objective": "buildable_objective(ensemble=8, burden_capacity=500)",
            "pass_threshold": PASS,
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "arms": arms,
        "paired_gnn_minus_mlp": paired,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
