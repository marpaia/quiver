"""Re-benchmark (W4 under grounding): does the MLP still out-amortize the GNN when designs are grounded?

The prior W4 found the feature-summary MLP the decisively stronger amortizer: one target-conditioned policy,
trained across a target distribution, designs held-out targets better than the graph policy, because message
passing dilutes the target-conditioning signal (the graph policy under-conditions). This repeats the
head-to-head with both conditioned policies *Component-grounded* on the merged Cello + ECF library, to check
whether grounding (and the activators it adds) changes that verdict. Grounding is orthogonal to the
conditioning bottleneck, so the expectation is that the MLP's amortization edge persists; this measures it.

For each arm (conditioned MLP vs conditioned GNN) trained once on the training targets, we report the mean
distinct grounded topologies per target on the trained split and the held-out split (the amortization test),
plus activator usage. A grounded design is scored directly (real kinetics, no tuning).

Run: ``uv run python grounded_amortization.py``. Scale with ``SEEDS=``, ``BUDGET=``, ``COND_STEPS=``,
``TRAIN_BATCH=``, ``BETA=``, ``LIBRARY=``, ``PASS=``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402

from _common import SENSOR_TARGET_GRID as TARGETS  # noqa: E402
from _common import env_float, env_int  # noqa: E402

from quiver.component import ComponentLibrary, library_from_name  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import amortized, gnn  # noqa: E402
from quiver.objective import TargetResponse, buildable_objective, dose_response_match  # noqa: E402
from quiver.research.eval import paired_summary  # noqa: E402

SEEDS = env_int("SEEDS", 3)
BUDGET = env_int("BUDGET", 48)
COND_STEPS = env_int("COND_STEPS", 40)
BATCH = env_int("TRAIN_BATCH", 8)
BETA = env_float("BETA", 4.0)
MAX_STEPS = env_int("MAX_STEPS", 8)
PASS = env_float("PASS", 0.3)
INDUCER = os.environ.get("INDUCER", "aTc")
LIBRARY_NAME = os.environ.get("LIBRARY", "cello+ecf")

HELD_OUT = {2, 5, 8, 11}  # interpolation split, as in amortization_gnn.py
TRAIN_TARGETS = [t for i, t in enumerate(TARGETS) if i not in HELD_OUT]
HELD_TARGETS = [t for i, t in enumerate(TARGETS) if i in HELD_OUT]

ARMS = {"mlp": amortized, "gnn": gnn}
RESULTS = Path(__file__).parent / "results" / "grounded_amortization.json"


def _activator_frac(grns: list[GRN], library: ComponentLibrary) -> float:
    def has_activator(grn: GRN) -> bool:
        for node in grn.nodes:
            if node.component_id is not None:
                try:
                    if library.by_id(node.component_id).role == "activator":
                        return True
                except KeyError:
                    continue
        return False

    return mean(1.0 if has_activator(g) else 0.0 for g in grns) if grns else 0.0


def _distinct_and_activators(
    module: object, policy: object, target: TargetResponse, library: ComponentLibrary, seed: int
) -> tuple[int, float]:
    """Distinct grounded topologies clearing PASS, and activator usage, for one target."""
    objective = buildable_objective(dose_response_match(target))
    grns = module.sample_conditioned(  # type: ignore[attr-defined]
        policy, target, BUDGET, max_steps=MAX_STEPS, seed=1000 + seed, library=library, inducer=INDUCER
    )
    passers = [g for g in grns if objective(g) >= PASS]
    return len({g.topology_hash() for g in passers}), _activator_frac(passers, library)


def main() -> None:
    library = library_from_name(LIBRARY_NAME)
    if library is None:
        raise SystemExit(f"unknown library {LIBRARY_NAME!r}")
    print(
        f"grounded W4 | library={LIBRARY_NAME} ({len(library)} components) | {SEEDS} seeds | β={BETA} | "
        f"budget {BUDGET} | cond train {COND_STEPS * BATCH} evals | "
        f"{len(TRAIN_TARGETS)} trained + {len(HELD_TARGETS)} held-out targets\n"
    )

    # per_seed[arm] -> list of {trained_distinct, held_distinct, activator_frac} per seed.
    per_seed: dict[str, list[dict[str, float]]] = {a: [] for a in ARMS}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for arm, module in ARMS.items():
            policy = module.train_conditioned(  # type: ignore[attr-defined]
                TRAIN_TARGETS,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                library=library,
                inducer=INDUCER,
            )
            trained = [_distinct_and_activators(module, policy, t, library, seed) for t in TRAIN_TARGETS]
            held = [_distinct_and_activators(module, policy, t, library, seed) for t in HELD_TARGETS]
            per_seed[arm].append(
                {
                    "trained_distinct": mean(d for d, _ in trained),
                    "held_distinct": mean(d for d, _ in held),
                    "activator_frac": mean(a for *_, a in trained + held),
                }
            )
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s)")

    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = per_seed[arm]
        arms[arm] = {
            "trained_distinct_mean": mean(r["trained_distinct"] for r in rows),
            "held_distinct_mean": mean(r["held_distinct"] for r in rows),
            "held_distinct_std": pstdev(r["held_distinct"] for r in rows),
            "activator_frac_mean": mean(r["activator_frac"] for r in rows),
        }

    print("trained-target distinct | held-out distinct | activator-frac, per arm:")
    for arm in ARMS:
        m = arms[arm]
        print(
            f"  {arm:4s} trained {m['trained_distinct_mean']:5.1f} | "
            f"held {m['held_distinct_mean']:5.1f} ± {m['held_distinct_std']:4.1f} | "
            f"act {m['activator_frac_mean']:.2f}"
        )

    gnn_held = [r["held_distinct"] for r in per_seed["gnn"]]
    mlp_held = [r["held_distinct"] for r in per_seed["mlp"]]
    s = paired_summary(gnn_held, mlp_held)
    print(
        f"\n== paired gnn - mlp (held-out distinct, per seed) ==\n"
        f"  {s.mean_diff:+.2f} [95% CI {s.ci_low:+.2f}, {s.ci_high:+.2f}] | win {s.win_rate:.0%} | "
        f"sign-test p={s.sign_p:.2g} | n={s.n}"
    )

    result = {
        "config": {
            "library": LIBRARY_NAME,
            "components": len(library),
            "seeds": SEEDS,
            "beta": BETA,
            "budget": BUDGET,
            "cond_train_evals": COND_STEPS * BATCH,
            "pass_threshold": PASS,
            "trained_targets": len(TRAIN_TARGETS),
            "held_out_targets": len(HELD_TARGETS),
        },
        "arms": arms,
        "paired_gnn_minus_mlp_held": {
            "mean_diff": s.mean_diff,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "win_rate": s.win_rate,
            "sign_p": s.sign_p,
            "n": s.n,
        },
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
