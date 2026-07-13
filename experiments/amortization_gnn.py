"""W4: does amortization carried by the physically-grounded (graph) policy hold, and beat the MLP?

Extends the amortization isolation study (``amortization_isolation.py``) to the graph policy. It asks two
things. First, the same controlled amortization question, now for the conditioned GNN: on trained targets,
does one shared policy cost anything versus per-target GNN specialists (the amortization gap)? On held-out
targets, does the conditioned GNN beat reusing the nearest trained target's specialist (the test that
separates a learned target-to-design map from interpolation by proximity)? Second, a head-to-head of the
conditioned graph policy against the conditioned feature-summary MLP (``amortized.py``), on both splits, to
see whether reading circuit physics improves the amortized target-to-design map.

It also reports the amortization crossover: distinct designs discovered per 1000 total simulations, counting
the conditioned policy's one-time training against the per-target specialists' repeated training, as the
number of targets served grows.

Run: ``uv run python experiments/amortization_gnn.py``. ``SPLIT=interpolation|extrapolation``,
``SEEDS=``, ``ENTROPY_COEF=`` (GNN entropy regularizer).
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence LOICA's per-simulation progress bars

import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, pstdev  # noqa: E402

from _common import SENSOR_TARGET_GRID as TARGETS  # noqa: E402
from _common import distinct, env_float, env_int  # noqa: E402

from quiver.eval import PairedSummary, paired_summary  # noqa: E402
from quiver.generate import tune_score  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import amortized, gnn  # noqa: E402
from quiver.learn.amortized import target_features  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402

SEEDS = env_int("SEEDS", 3)
EVAL_BUDGET = env_int("EVAL_BUDGET", 300)
TUNE_STEPS = env_int("TUNE_STEPS", 2)
COND_STEPS = env_int("COND_STEPS", 80)
SPEC_STEPS = env_int("SPEC_STEPS", 40)
BATCH = env_int("TRAIN_BATCH", 12)
BETA = env_float("BETA", 4.0)
MAX_STEPS = env_int("MAX_STEPS", 8)
ENTROPY_COEF = env_float("ENTROPY_COEF", 0.0)

SPLIT = os.environ.get("SPLIT", "interpolation")
HELD_OUT = {2, 5, 8, 11} if SPLIT == "interpolation" else {8, 9, 10, 11}
TRAIN_INDICES = [i for i in range(len(TARGETS)) if i not in HELD_OUT]

# An optional filename tag, so a re-run under a different training regime does not clobber the primary run.
TAG = os.environ.get("TAG", "")
RESULTS = Path(__file__).parent / "results" / f"amortization_gnn_{SPLIT}{TAG}.json"


def score_terminals(terminals: list[GRN], target: TargetResponse, seed: int) -> int:
    """Distinct high-scoring topologies among sampled terminals, scored against ``target``."""
    behavior = dose_response_match(target)
    rng = random.Random(seed)
    return distinct([(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS)) for g in terminals])


def nearest_trained(held_index: int) -> int:
    """The trained-target index closest to a held-out target in target-feature space."""
    hf = target_features(TARGETS[held_index])
    return min(TRAIN_INDICES, key=lambda j: sum((a - b) ** 2 for a, b in zip(hf, target_features(TARGETS[j]))))


def _paired(s: PairedSummary) -> dict[str, float]:
    return {
        "mean_diff": s.mean_diff,
        "ci_low": s.ci_low,
        "ci_high": s.ci_high,
        "win_rate": s.win_rate,
        "sign_p": s.sign_p,
        "n": s.n,
    }


def main() -> None:
    n_samples = EVAL_BUDGET // (TUNE_STEPS + 1)
    print(
        f"SPLIT={SPLIT} | {SEEDS} seeds | {len(TARGETS)} targets ({len(HELD_OUT)} held out) "
        f"| {n_samples} samples/target | β={BETA} | GNN entropy_coef={ENTROPY_COEF}\n"
    )

    # Aligned per-(seed, target) cells.
    tr_gnn: list[int] = []  # conditioned GNN, trained targets
    tr_spec: list[int] = []  # per-target GNN specialist, trained targets
    tr_mlp: list[int] = []  # conditioned MLP, trained targets
    ho_gnn: list[int] = []  # conditioned GNN, held-out (zero-shot)
    ho_near: list[int] = []  # nearest trained GNN specialist reused, held-out
    ho_mlp: list[int] = []  # conditioned MLP, held-out (zero-shot)

    t0 = time.perf_counter()
    for seed in range(SEEDS):
        train_targets = [TARGETS[i] for i in TRAIN_INDICES]
        cond_gnn = gnn.train_conditioned(
            train_targets,
            steps=COND_STEPS,
            batch=BATCH,
            beta=BETA,
            max_steps=MAX_STEPS,
            seed=seed,
            entropy_coef=ENTROPY_COEF,
        )
        cond_mlp = amortized.train_conditioned(
            train_targets, steps=COND_STEPS, batch=BATCH, beta=BETA, max_steps=MAX_STEPS, seed=seed
        )
        specialists = {
            i: gnn.train(
                dose_response_match(TARGETS[i]),
                steps=SPEC_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
                entropy_coef=ENTROPY_COEF,
            )
            for i in TRAIN_INDICES
        }

        for i in TRAIN_INDICES:
            g = gnn.sample_conditioned(cond_gnn, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=3000 + seed)
            m = amortized.sample_conditioned(cond_mlp, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=3000 + seed)
            s = gnn.sample_terminals(specialists[i], n_samples, max_steps=MAX_STEPS, seed=3000 + seed)
            tr_gnn.append(score_terminals(g, TARGETS[i], 3000 + seed))
            tr_mlp.append(score_terminals(m, TARGETS[i], 3000 + seed))
            tr_spec.append(score_terminals(s, TARGETS[i], 3000 + seed))

        for i in sorted(HELD_OUT):
            donor = specialists[nearest_trained(i)]
            g = gnn.sample_conditioned(cond_gnn, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=4000 + seed)
            m = amortized.sample_conditioned(cond_mlp, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=4000 + seed)
            near = gnn.sample_terminals(donor, n_samples, max_steps=MAX_STEPS, seed=4000 + seed)
            ho_gnn.append(score_terminals(g, TARGETS[i], 4000 + seed))
            ho_mlp.append(score_terminals(m, TARGETS[i], 4000 + seed))
            ho_near.append(score_terminals(near, TARGETS[i], 4000 + seed))

        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    print(f"\nran {SEEDS} seeds in {time.perf_counter() - t0:.0f}s\n")

    print("== trained targets ==")
    print(
        f"  conditioned-GNN {mean(tr_gnn):.1f} ± {pstdev(tr_gnn):.1f} | "
        f"GNN specialist {mean(tr_spec):.1f} ± {pstdev(tr_spec):.1f} | "
        f"conditioned-MLP {mean(tr_mlp):.1f} ± {pstdev(tr_mlp):.1f}"
    )
    tr_gap = paired_summary(tr_gnn, tr_spec)
    tr_vs_mlp = paired_summary(tr_gnn, tr_mlp)
    print(
        f"  gap (cond-GNN − specialist):  {tr_gap.mean_diff:+.2f} [{tr_gap.ci_low:+.2f}, {tr_gap.ci_high:+.2f}]"
        f" | win {tr_gap.win_rate:.0%} | p={tr_gap.sign_p:.2g}"
    )
    print(
        f"  cond-GNN − cond-MLP:          {tr_vs_mlp.mean_diff:+.2f} "
        f"[{tr_vs_mlp.ci_low:+.2f}, {tr_vs_mlp.ci_high:+.2f}] | win {tr_vs_mlp.win_rate:.0%} | p={tr_vs_mlp.sign_p:.2g}"
    )

    print("\n== held-out targets (zero-shot) ==")
    print(
        f"  conditioned-GNN {mean(ho_gnn):.1f} ± {pstdev(ho_gnn):.1f} | "
        f"nearest-reuse {mean(ho_near):.1f} ± {pstdev(ho_near):.1f} | "
        f"conditioned-MLP {mean(ho_mlp):.1f} ± {pstdev(ho_mlp):.1f}"
    )
    ho_vs_near = paired_summary(ho_gnn, ho_near)
    ho_vs_mlp = paired_summary(ho_gnn, ho_mlp)
    print(
        f"  cond-GNN − nearest-reuse:     {ho_vs_near.mean_diff:+.2f} "
        f"[{ho_vs_near.ci_low:+.2f}, {ho_vs_near.ci_high:+.2f}] | win {ho_vs_near.win_rate:.0%} "
        f"| p={ho_vs_near.sign_p:.2g}"
    )
    print(
        f"  cond-GNN − cond-MLP:          {ho_vs_mlp.mean_diff:+.2f} "
        f"[{ho_vs_mlp.ci_low:+.2f}, {ho_vs_mlp.ci_high:+.2f}] | win {ho_vs_mlp.win_rate:.0%} | p={ho_vs_mlp.sign_p:.2g}"
    )

    # Amortization crossover: distinct designs per 1000 total sims as the number of targets K grows.
    # Conditioned: one training (COND_STEPS*BATCH) + K*EVAL_BUDGET. Specialists: K*(SPEC_STEPS*BATCH + EVAL_BUDGET).
    n_targets = len(TARGETS)
    per_target_gnn = mean(tr_gnn + ho_gnn)
    per_target_spec = mean(tr_spec)  # specialists only exist on trained targets
    cond_train = COND_STEPS * BATCH
    spec_train = SPEC_STEPS * BATCH
    crossover: dict[str, dict[str, float]] = {}
    for k in (1, 2, 4, 8, n_targets):
        cond_sims = cond_train + k * EVAL_BUDGET
        spec_sims = k * (spec_train + EVAL_BUDGET)
        crossover[str(k)] = {
            "conditioned_per_1k": 1000 * k * per_target_gnn / cond_sims,
            "specialist_per_1k": 1000 * k * per_target_spec / spec_sims,
        }
    print("\n== crossover: distinct designs per 1000 total sims (training counted) ==")
    for label, row in crossover.items():
        print(
            f"  K={label:>2}: conditioned-GNN {row['conditioned_per_1k']:.1f} "
            f"| specialists {row['specialist_per_1k']:.1f}"
        )

    result = {
        "config": {
            "split": SPLIT,
            "seeds": SEEDS,
            "eval_budget": EVAL_BUDGET,
            "tune_steps": TUNE_STEPS,
            "cond_sims": cond_train,
            "spec_sims": spec_train,
            "beta": BETA,
            "entropy_coef": ENTROPY_COEF,
            "held_out": sorted(HELD_OUT),
        },
        "trained": {
            "conditioned_gnn_mean": mean(tr_gnn),
            "gnn_specialist_mean": mean(tr_spec),
            "conditioned_mlp_mean": mean(tr_mlp),
            "paired_gnn_minus_specialist": _paired(tr_gap),
            "paired_gnn_minus_mlp": _paired(tr_vs_mlp),
        },
        "held_out": {
            "conditioned_gnn_mean": mean(ho_gnn),
            "nearest_reuse_mean": mean(ho_near),
            "conditioned_mlp_mean": mean(ho_mlp),
            "paired_gnn_minus_nearest": _paired(ho_vs_near),
            "paired_gnn_minus_mlp": _paired(ho_vs_mlp),
        },
        "crossover_per_1k_sims": crossover,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
