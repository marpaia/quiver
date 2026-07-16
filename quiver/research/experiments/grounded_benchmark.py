"""Re-benchmark (W1 under grounding): GNN vs MLP when designs are built from real SBOL Components.

The prior W1 compared the structured-env feature-summary MLP (:mod:`quiver.learn.structured`) against the
graph policy (:mod:`quiver.learn.gnn`) over abstract circuits. This repeats that head-to-head with both
policies *Component-grounded* on the merged Cello + ECF library, so each stage is a real repressor or
activator and a design can alternate activation and repression. Both arms search the identical action space
at identical budget and seeds and select Components from the identical library; only the policy representation
differs.

Reported per arm: **breadth** (distinct topologies, and distinct grounded assignments), **quality** (mean and
peak buildability-aware score), and the grounding-specific **activator usage** (fraction of delivered designs
that use at least one activator, and mean activators per design) --- the axis that only exists once the
library has activators. A grounded design's kinetics are the real part values, so it is scored directly (no
parameter tuning).

Run: ``uv run python grounded_benchmark.py``. Scale with ``SEEDS=``, ``BETAS=``, ``BUDGET=``, ``TRAIN_STEPS=``,
``TRAIN_BATCH=``, ``LIBRARY=`` (default ``cello+ecf``), ``PASS=``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402

from _common import env_float, env_int  # noqa: E402

from quiver.component import ComponentLibrary, library_from_name  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import gnn, structured  # noqa: E402
from quiver.objective import TargetResponse, buildable_objective, dose_response_match  # noqa: E402
from quiver.research.eval import paired_summary  # noqa: E402

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")

SEEDS = env_int("SEEDS", 3)
BETAS = [float(b) for b in os.environ.get("BETAS", "1").split(",") if b.strip()]
BUDGET = env_int("BUDGET", 80)
TRAIN_STEPS = env_int("TRAIN_STEPS", 20)
TRAIN_BATCH = env_int("TRAIN_BATCH", 8)
MAX_STEPS = env_int("MAX_STEPS", 8)
PASS = env_float("PASS", 0.3)  # buildable scores sit lower than shape-match; a modest high-scoring cut
INDUCER = os.environ.get("INDUCER", "aTc")
LIBRARY_NAME = os.environ.get("LIBRARY", "cello+ecf")

ARMS = {"mlp": structured, "gnn": gnn}
RESULTS = Path(__file__).parent / "results" / "grounded_benchmark.json"


def _roles(grn: GRN, library: ComponentLibrary) -> list[str]:
    roles = []
    for node in grn.nodes:
        if node.component_id is not None:
            try:
                roles.append(library.by_id(node.component_id).role)
            except KeyError:
                continue
    return roles


def summarize(scored: list[tuple[GRN, float]], library: ComponentLibrary) -> dict[str, float]:
    """Breadth, quality, and activator usage of a delivered grounded set."""
    passers = [(g, s) for g, s in scored if s >= PASS]
    topologies = {g.topology_hash() for g, _ in passers}
    assignments = {g.structural_hash() for g, _ in passers}
    scores = [s for _, s in scored]
    act_counts = [sum(1 for r in _roles(g, library) if r == "activator") for g, _ in passers]
    return {
        "distinct": len(topologies),
        "distinct_assignments": len(assignments),
        "passers": len(passers),
        "peak_score": max(scores) if scores else 0.0,
        "mean_score": mean(scores) if scores else 0.0,
        "activator_frac": (mean(1.0 if c > 0 else 0.0 for c in act_counts) if act_counts else 0.0),
        "activators_per_design": (mean(act_counts) if act_counts else 0.0),
    }


def deliver(module: object, behavior: object, objective: object, library: ComponentLibrary, beta: float, seed: int):
    """Train a grounded policy and score its sampled designs directly (real kinetics, no tuning)."""
    policy = module.train(  # type: ignore[attr-defined]
        behavior,
        steps=TRAIN_STEPS,
        batch=TRAIN_BATCH,
        beta=beta,
        max_steps=MAX_STEPS,
        seed=seed,
        objective=objective,
        library=library,
        inducer=INDUCER,
    )
    terminals = module.sample_terminals(  # type: ignore[attr-defined]
        policy, BUDGET, max_steps=MAX_STEPS, seed=1000 + seed, library=library, inducer=INDUCER
    )
    return [(grn, objective(grn)) for grn in terminals]  # type: ignore[operator]


def main() -> None:
    library = library_from_name(LIBRARY_NAME)
    if library is None:
        raise SystemExit(f"unknown library {LIBRARY_NAME!r}")
    behavior = dose_response_match(TARGET)
    objective = buildable_objective(behavior)
    print(
        f"grounded W1 | library={LIBRARY_NAME} ({len(library)} components: "
        f"{len(library.repressors())}R+{len(library.activators())}A+{len(library.sensors())}S) | "
        f"{SEEDS} seeds | β ∈ {BETAS} | budget {BUDGET} | train {TRAIN_STEPS * TRAIN_BATCH} evals (ensemble 8)\n"
    )

    per_seed: dict[str, dict[float, list[dict[str, float]]]] = {a: {b: [] for b in BETAS} for a in ARMS}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for beta in BETAS:
            for arm, module in ARMS.items():
                scored = deliver(module, behavior, objective, library, beta, seed)
                per_seed[arm][beta].append(summarize(scored, library))
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s)")

    arms: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARMS:
        arms[arm] = {}
        for beta in BETAS:
            rows = per_seed[arm][beta]
            distinct = [r["distinct"] for r in rows]
            arms[arm][f"beta={beta}"] = {
                "distinct_mean": mean(distinct),
                "distinct_std": pstdev(distinct),
                "distinct_assignments_mean": mean(r["distinct_assignments"] for r in rows),
                "peak_mean": mean(r["peak_score"] for r in rows),
                "meanscore_mean": mean(r["mean_score"] for r in rows),
                "activator_frac_mean": mean(r["activator_frac"] for r in rows),
                "activators_per_design_mean": mean(r["activators_per_design"] for r in rows),
                "per_seed_distinct": distinct,
            }

    print("\ndistinct topo | assigns | peak | mean | activator-frac | acts/design, per arm × β:")
    for beta in BETAS:
        print(f"  β = {beta}")
        for arm in ARMS:
            m = arms[arm][f"beta={beta}"]
            print(
                f"    {arm:4s} {m['distinct_mean']:5.1f} ± {m['distinct_std']:4.1f} | "
                f"assign {m['distinct_assignments_mean']:5.1f} | peak {m['peak_mean']:.3f} | "
                f"mean {m['meanscore_mean']:.3f} | act {m['activator_frac_mean']:.2f} | "
                f"a/d {m['activators_per_design_mean']:.2f}"
            )

    print("\n== paired gnn - mlp (distinct topologies, per seed) ==")
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
            "library": LIBRARY_NAME,
            "components": len(library),
            "repressors": len(library.repressors()),
            "activators": len(library.activators()),
            "sensors": len(library.sensors()),
            "seeds": SEEDS,
            "betas": BETAS,
            "budget": BUDGET,
            "train_evals": TRAIN_STEPS * TRAIN_BATCH,
            "objective": "buildable_objective(ensemble=8, burden_capacity=500)",
            "pass_threshold": PASS,
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "arms": arms,
        "paired_gnn_minus_mlp": paired,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
