"""W2: what does the buildability-aware objective do to the delivered portfolio?

Holds the architecture and the sampler fixed (the feature-summary policy over the structured action space)
and varies only the *training objective*, then reads the delivered portfolio on a common shape-match
yardstick. Three arms:

- **shape**: the bare dose-response shape-match (``score_once``), the objective a design tool optimizes by
  default.
- **robustness**: robustness over a parameter ensemble, burden disabled
  (``buildable_objective(burden_capacity=None)``).
- **buildable**: robustness discounted by metabolic burden (the default ``buildable_objective``).

Every arm is tuned and scored against the same shape-match yardstick, so breadth is comparable; the question
is whether the buildability-aware training signal removes the deep-cascade / fit-on-paper pathology, i.e.
delivers shallower, lower-burden portfolios at comparable shape-match breadth. Reported per arm: breadth,
quality, mean burden, cascade-depth (regulator-count) distribution, and realizable-parameter fraction.

``ENV`` selects the environment. The default ``base`` uses the base-env GFlowNet
(:mod:`quiver.learn.gflownet`), whose relay move has no regulator cap, so cascade depth is free to vary and
the burden term can discriminate. ``structured`` uses the regulator-capped structured MLP
(:mod:`quiver.learn.structured`); there the cap already bounds depth, so the objective has little room to
move burden, which is itself worth recording.

The burden-in-objective result is a *buildability* study, reported here separately from the decorrelation
hedge, which keeps burden a held-out failure model (robustness-only objective). Folding burden into the
objective would make it no longer held-out there.

Run: ``uv run python experiments/objective_ablation.py``. Scale with ``SEEDS=`` and ``BETA=``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402

from _common import BETA, EVAL_BUDGET, MAX_STEPS, PASS, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS, env_int  # noqa: E402
from _policy import deliver, delivered_portfolio, regulator_count, summarize  # noqa: E402

from quiver.eval import paired_summary  # noqa: E402
from quiver.learn import gflownet, structured  # noqa: E402
from quiver.objective import (  # noqa: E402
    Behavior,
    ObjectiveFn,
    TargetResponse,
    buildable_objective,
    dose_response_match,
)

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")

SEEDS = env_int("SEEDS", 3)
N_SAMPLES = EVAL_BUDGET // (TUNE_STEPS + 1)
ENVS = {"base": gflownet, "structured": structured}
ENV = os.environ.get("ENV", "base")
MODULE = ENVS[ENV]
RESULTS = Path(__file__).parent / "results" / f"objective_ablation_{ENV}.json"

BUILD_KEYS = ("burden_mean", "regulators_mean", "realizable_frac")


def objectives(behavior: Behavior) -> dict[str, ObjectiveFn | None]:
    """The three training objectives compared, keyed by arm name (``None`` is the shape-match)."""
    return {
        "shape": None,
        "robustness": buildable_objective(behavior, burden_capacity=None),
        "buildable": buildable_objective(behavior),
    }


def main() -> None:
    behavior = dose_response_match(TARGET)
    arms = objectives(behavior)
    print(
        f"{SEEDS} seeds | β={BETA} | deliver {N_SAMPLES} samples (tune {TUNE_STEPS}) "
        f"| env={ENV} ({MODULE.__name__}) | arms: {list(arms)}\n"
    )

    per_seed: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    depths: dict[str, Counter[int]] = {a: Counter() for a in arms}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for arm, objective in arms.items():
            designs = deliver(
                MODULE,
                behavior,
                objective=objective,
                beta=BETA,
                seed=seed,
                n_samples=N_SAMPLES,
                tune_steps=TUNE_STEPS,
                train_steps=TRAIN_STEPS,
                train_batch=TRAIN_BATCH,
                max_steps=MAX_STEPS,
            )
            per_seed[arm].append(summarize(designs, pass_threshold=PASS))
            depths[arm].update(regulator_count(grn) for grn, _ in delivered_portfolio(designs, PASS))
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    summary: dict[str, dict[str, object]] = {}
    for arm in arms:
        rows = per_seed[arm]
        distinct = [r["distinct"] for r in rows]
        summary[arm] = {
            "distinct_mean": mean(distinct),
            "distinct_std": pstdev(distinct),
            "peak_mean": mean(r["peak_score"] for r in rows),
            "meanscore_mean": mean(r["mean_score"] for r in rows),
            **{k: mean(r[k] for r in rows) for k in BUILD_KEYS},
            "depth_histogram": {str(d): depths[arm][d] for d in sorted(depths[arm])},
            "per_seed_distinct": distinct,
            "per_seed_regulators": [r["regulators_mean"] for r in rows],
            "per_seed_burden": [r["burden_mean"] for r in rows],
        }

    print("\ndistinct high-scoring | peak | mean | burden | regs | realizable | depth histogram:")
    for arm in arms:
        m = summary[arm]
        print(
            f"  {arm:11s} {m['distinct_mean']:5.1f} ± {m['distinct_std']:4.1f} | "
            f"peak {m['peak_mean']:.3f} | mean {m['meanscore_mean']:.3f} | "
            f"burden {m['burden_mean']:.3f} | regs {m['regulators_mean']:.2f} | "
            f"real {m['realizable_frac']:.2f} | depth {m['depth_histogram']}"
        )

    # The buildability payoff: does the buildable objective deliver a shallower, lower-burden portfolio than
    # the bare shape-match? Paired per-seed (positive = buildable is shallower / lower-burden).
    paired: dict[str, dict[str, float]] = {}
    print("\n== paired shape − buildable (positive = buildable shallower / lower burden) ==")
    for axis, key in (("regulators", "per_seed_regulators"), ("burden", "per_seed_burden")):
        shape_vals = summary["shape"][key]
        build_vals = summary["buildable"][key]
        s = paired_summary(shape_vals, build_vals)  # type: ignore[arg-type]
        print(
            f"  {axis:11s} {s.mean_diff:+.3f} [95% CI {s.ci_low:+.3f}, {s.ci_high:+.3f}] "
            f"| win {s.win_rate:.0%} | sign-test p={s.sign_p:.2g} | n={s.n}"
        )
        paired[axis] = {
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
            "beta": BETA,
            "deliver_samples": N_SAMPLES,
            "tune_steps": TUNE_STEPS,
            "train_evals": TRAIN_STEPS * TRAIN_BATCH,
            "env": ENV,
            "policy": MODULE.__name__,
            "pass_threshold": PASS,
            "yardstick": "score_once shape-match (tuning and pass gate)",
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "arms": summary,
        "paired_shape_minus_buildable": paired,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
