"""Diagnose why the conditioned graph policy loses to the feature-summary MLP at amortization.

W4 found the conditioned GNN a decisively weaker amortized policy than the MLP. The suspected cause is the
conditioning scheme: the GNN appends the target encoding to every node's *input* features, and directed
message passing (mean-aggregation over neighbors) mixes that constant, structure-independent signal into the
structural embedding, diluting it by the time the action heads read it. The MLP, with a flat feature vector,
keeps the target salient.

This tests that hypothesis and a fix. It compares three amortized policies -- the feature-summary MLP, the
node-conditioned GNN (target in node features), and the head-conditioned GNN (target fed to the action heads,
undiluted) -- on two axes:

- **Target sensitivity:** how much a policy's sampled topology distribution changes across targets, measured
  as the mean pairwise Jaccard overlap of the sampled topology sets over the target grid (no tuning, no
  simulation). A policy that ignores the target samples the same topologies everywhere (overlap near 1); a
  target-sensitive policy has lower overlap. If node-conditioning is diluted, its overlap is high and
  head-conditioning lowers it.
- **Amortization:** distinct high-scoring topologies on trained and held-out targets, paired against the MLP,
  to see whether head-conditioning closes the gap W4 measured.

Run: ``uv run python experiments/gnn_conditioning.py``. ``SEEDS=``, ``SPLIT=``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402
from typing import Any, Callable  # noqa: E402

from _common import SENSOR_TARGET_GRID as TARGETS  # noqa: E402
from _common import distinct, env_int  # noqa: E402

from quiver.generate import tune_score  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import amortized, gnn  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402
from quiver.research.eval import paired_summary  # noqa: E402

SEEDS = env_int("SEEDS", 3)
EVAL_BUDGET = env_int("EVAL_BUDGET", 300)
TUNE_STEPS = env_int("TUNE_STEPS", 2)
COND_STEPS = env_int("COND_STEPS", 80)
BATCH = env_int("TRAIN_BATCH", 12)
BETA = env_int("BETA", 4)
MAX_STEPS = env_int("MAX_STEPS", 8)
SENS_SAMPLES = env_int("SENS_SAMPLES", 50)

SPLIT = os.environ.get("SPLIT", "interpolation")
HELD_OUT = {2, 5, 8, 11} if SPLIT == "interpolation" else {8, 9, 10, 11}
TRAIN_INDICES = [i for i in range(len(TARGETS)) if i not in HELD_OUT]

RESULTS = Path(__file__).parent / "results" / "gnn_conditioning.json"


def score_terminals(terminals: list[GRN], target: TargetResponse, seed: int) -> int:
    behavior = dose_response_match(target)
    rng = random.Random(seed)
    return distinct([(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS)) for g in terminals])


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def target_sensitivity(sampler: Callable[[TargetResponse, int], list[GRN]]) -> float:
    """Mean pairwise Jaccard overlap of sampled topology sets across the target grid (no tuning).

    ``sampler(target, seed)`` returns sampled circuits for a target. High overlap ⇒ the policy samples the
    same topologies regardless of target (it ignores the conditioning); low overlap ⇒ target-sensitive."""
    sets = [{g.topology_hash() for g in sampler(TARGETS[i], 7000 + i)} for i in range(len(TARGETS))]
    overlaps = [_jaccard(a, b) for a, b in combinations(sets, 2)]
    return mean(overlaps) if overlaps else 1.0


def main() -> None:
    n_samples = EVAL_BUDGET // (TUNE_STEPS + 1)
    print(
        f"SPLIT={SPLIT} | {SEEDS} seeds | β={BETA} | {n_samples} samples/target | "
        f"sensitivity over {SENS_SAMPLES} samples/target\n"
    )
    policies = ("mlp", "node-gnn", "head-gnn")
    sens: dict[str, list[float]] = {p: [] for p in policies}
    tr: dict[str, list[int]] = {p: [] for p in policies}
    ho: dict[str, list[int]] = {p: [] for p in policies}

    t0 = time.perf_counter()
    for seed in range(SEEDS):
        train_targets = [TARGETS[i] for i in TRAIN_INDICES]
        trained = {
            "mlp": amortized.train_conditioned(
                train_targets, steps=COND_STEPS, batch=BATCH, beta=BETA, max_steps=MAX_STEPS, seed=seed
            ),
            "node-gnn": gnn.train_conditioned(
                train_targets,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                cond_mode="node",
            ),
            "head-gnn": gnn.train_conditioned(
                train_targets,
                steps=COND_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                cond_mode="head",
            ),
        }

        def sampler_for(name: str, policy: Any) -> Callable[[TargetResponse, int], list[GRN]]:
            if name == "mlp":
                return lambda t, s: amortized.sample_conditioned(policy, t, SENS_SAMPLES, max_steps=MAX_STEPS, seed=s)
            mode = "node" if name == "node-gnn" else "head"
            return lambda t, s: gnn.sample_conditioned(
                policy, t, SENS_SAMPLES, max_steps=MAX_STEPS, seed=s, cond_mode=mode
            )

        for name in policies:
            sens[name].append(target_sensitivity(sampler_for(name, trained[name])))

        def eval_sampler(name: str, policy: Any, target: TargetResponse, seed_off: int) -> int:
            if name == "mlp":
                terms = amortized.sample_conditioned(policy, target, n_samples, max_steps=MAX_STEPS, seed=seed_off)
            else:
                mode = "node" if name == "node-gnn" else "head"
                terms = gnn.sample_conditioned(
                    policy, target, n_samples, max_steps=MAX_STEPS, seed=seed_off, cond_mode=mode
                )
            return score_terminals(terms, target, seed_off)

        for i in TRAIN_INDICES:
            for name in policies:
                tr[name].append(eval_sampler(name, trained[name], TARGETS[i], 3000 + seed))
        for i in sorted(HELD_OUT):
            for name in policies:
                ho[name].append(eval_sampler(name, trained[name], TARGETS[i], 4000 + seed))

        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    print("\n== target sensitivity (mean cross-target topology overlap; lower = more target-sensitive) ==")
    for name in policies:
        print(f"  {name:9s} {mean(sens[name]):.3f} ± {pstdev(sens[name]):.3f}")

    print("\n== amortization (distinct high-scoring topologies) ==")
    for name in policies:
        print(
            f"  {name:9s} trained {mean(tr[name]):5.1f} ± {pstdev(tr[name]):4.1f} | "
            f"held-out {mean(ho[name]):5.1f} ± {pstdev(ho[name]):4.1f}"
        )

    print("\n== paired vs MLP (positive = policy beats MLP) ==")
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("node-gnn", "head-gnn"):
        paired[name] = {}
        for split, data in (("trained", tr), ("held_out", ho)):
            s = paired_summary(data[name], data["mlp"])
            print(
                f"  {name:9s} {split:8s}: {s.mean_diff:+.2f} [{s.ci_low:+.2f}, {s.ci_high:+.2f}] "
                f"| win {s.win_rate:.0%} | p={s.sign_p:.2g}"
            )
            paired[name][split] = {
                "mean_diff": s.mean_diff,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "win_rate": s.win_rate,
                "sign_p": s.sign_p,
                "n": s.n,
            }

    result = {
        "config": {
            "split": SPLIT,
            "seeds": SEEDS,
            "beta": BETA,
            "eval_budget": EVAL_BUDGET,
            "sens_samples": SENS_SAMPLES,
            "cond_sims": COND_STEPS * BATCH,
        },
        "target_sensitivity": {p: {"mean": mean(sens[p]), "std": pstdev(sens[p])} for p in policies},
        "amortization": {p: {"trained_mean": mean(tr[p]), "held_out_mean": mean(ho[p])} for p in policies},
        "paired_vs_mlp": paired,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
