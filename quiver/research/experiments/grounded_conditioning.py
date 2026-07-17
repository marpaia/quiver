"""Does the grounded conditioned GNN actually condition on the target, or is its held-out breadth a mirage?

Grounded W4 (:mod:`grounded_amortization`) found the conditioned GNN delivers more distinct held-out designs
than the MLP --- reversing the abstract W4. But distinct-count rewards *breadth*, not target-specificity: a
policy that samples the same broad set for every target scores well without amortizing. The abstract-era
diagnosis of the GNN was exactly this failure --- cross-target topology overlap 0.81 (the node-conditioned GNN
ignored its target) versus the MLP's 0.001.

This ports that diagnostic to the grounded conditioned policies. **Target sensitivity** is the mean pairwise
Jaccard overlap of the sampled topology sets across the target grid (pure sampling, no simulation): overlap
near 1 means the policy samples the same topologies regardless of target (it ignores the conditioning); low
overlap means it is genuinely target-sensitive. If grounding lowered the GNN's overlap, its held-out breadth
reflects real conditioning; if the overlap is still high, the breadth is target-agnostic and the MLP remains
the true amortizer.

Run: ``uv run python grounded_conditioning.py``. Scale with ``SEEDS=``, ``COND_STEPS=``, ``TRAIN_BATCH=``,
``BETA=``, ``SENS_SAMPLES=``, ``LIBRARY=``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402
from typing import Callable  # noqa: E402

from _common import SENSOR_TARGET_GRID as TARGETS  # noqa: E402
from _common import env_float, env_int  # noqa: E402

from quiver.component import library_from_name  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import amortized, gnn  # noqa: E402
from quiver.objective import TargetResponse  # noqa: E402

SEEDS = env_int("SEEDS", 3)
COND_STEPS = env_int("COND_STEPS", 60)
BATCH = env_int("TRAIN_BATCH", 10)
BETA = env_float("BETA", 4.0)
MAX_STEPS = env_int("MAX_STEPS", 8)
SENS_SAMPLES = env_int("SENS_SAMPLES", 40)
INDUCER = os.environ.get("INDUCER", "aTc")
LIBRARY_NAME = os.environ.get("LIBRARY", "cello+ecf")

RESULTS = Path(__file__).parent / "results" / "grounded_conditioning.json"


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def target_sensitivity(sampler: Callable[[TargetResponse, int], list[GRN]]) -> float:
    """Mean pairwise Jaccard overlap of sampled topology sets across the target grid (no simulation)."""
    sets = [{g.topology_hash() for g in sampler(TARGETS[i], 7000 + i)} for i in range(len(TARGETS))]
    overlaps = [_jaccard(a, b) for a, b in combinations(sets, 2)]
    return mean(overlaps) if overlaps else 1.0


def main() -> None:
    library = library_from_name(LIBRARY_NAME)
    if library is None:
        raise SystemExit(f"unknown library {LIBRARY_NAME!r}")
    print(
        f"grounded conditioning | library={LIBRARY_NAME} ({len(library)} components) | {SEEDS} seeds | "
        f"β={BETA} | cond train {COND_STEPS * BATCH} evals | sensitivity over {SENS_SAMPLES} samples/target\n"
        "  (overlap near 1 = ignores target; low = target-sensitive; abstract: MLP 0.001, node-GNN 0.81)\n"
    )
    policies = ("mlp", "node-gnn", "head-gnn")
    sens: dict[str, list[float]] = {p: [] for p in policies}

    t0 = time.perf_counter()
    for seed in range(SEEDS):
        trained = {
            "mlp": amortized.train_conditioned(
                TARGETS,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                library=library,
                inducer=INDUCER,
            ),
            "node-gnn": gnn.train_conditioned(
                TARGETS,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                cond_mode="node",
                library=library,
                inducer=INDUCER,
            ),
            "head-gnn": gnn.train_conditioned(
                TARGETS,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                cond_mode="head",
                library=library,
                inducer=INDUCER,
            ),
        }

        def mlp_sampler(target: TargetResponse, s: int) -> list[GRN]:
            return amortized.sample_conditioned(
                trained["mlp"], target, SENS_SAMPLES, max_steps=MAX_STEPS, seed=s, library=library, inducer=INDUCER
            )

        def node_sampler(target: TargetResponse, s: int) -> list[GRN]:
            return gnn.sample_conditioned(
                trained["node-gnn"],
                target,
                SENS_SAMPLES,
                max_steps=MAX_STEPS,
                seed=s,
                cond_mode="node",
                library=library,
                inducer=INDUCER,
            )

        def head_sampler(target: TargetResponse, s: int) -> list[GRN]:
            return gnn.sample_conditioned(
                trained["head-gnn"],
                target,
                SENS_SAMPLES,
                max_steps=MAX_STEPS,
                seed=s,
                cond_mode="head",
                library=library,
                inducer=INDUCER,
            )

        sens["mlp"].append(target_sensitivity(mlp_sampler))
        sens["node-gnn"].append(target_sensitivity(node_sampler))
        sens["head-gnn"].append(target_sensitivity(head_sampler))
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s)")

    print("\ncross-target topology overlap (lower = more target-sensitive), per policy:")
    result: dict[str, dict[str, float]] = {}
    for p in policies:
        m, sd = mean(sens[p]), pstdev(sens[p])
        result[p] = {"overlap_mean": m, "overlap_std": sd, "per_seed": sens[p]}
        print(f"  {p:9s} {m:.3f} ± {sd:.3f}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "config": {
                    "library": LIBRARY_NAME,
                    "components": len(library),
                    "seeds": SEEDS,
                    "beta": BETA,
                    "cond_train_evals": COND_STEPS * BATCH,
                    "sensitivity_samples": SENS_SAMPLES,
                },
                "overlap": result,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
