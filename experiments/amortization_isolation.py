"""Isolating amortization from sampler quality.

The amortization comparison in ``amortization.py`` pits the one conditioned policy against per-target
random and evolutionary search, which cannot amortize by construction. That shows the conditioned policy
wins, but not *why*: a strong diversity-seeking sampler evaluated on more targets (H2) would win the same
way. This experiment adds the controls that separate amortization (H1) from sampler quality (H2):

- **per-target specialist**: a single-target GFlowNet trained on each trained target. Conditioned vs.
  specialist on the *trained* targets measures the amortization gap: does sharing one policy across
  targets cost anything relative to training one policy per target?
- **nearest-trained-target zero-shot**: on each held-out target, reuse the specialist of the *nearest
  trained target* unchanged. Conditioned vs. this control on held-out targets is the decisive test: if the
  conditioned policy only matched nearest-neighbor reuse, its "generalization" would be interpolation by
  proximity, not a learned target-to-design map.

Run: ``uv run python experiments/amortization_isolation.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from statistics import mean, pstdev

from _common import SENSOR_TARGET_GRID as TARGETS
from _common import distinct

from quiver.eval import PairedSummary, paired_summary
from quiver.generate import tune_score
from quiver.grn import GRN
from quiver.learn.amortized import sample_conditioned, target_features, train_conditioned
from quiver.learn.gflownet import PolicyNet, sample_terminals, train
from quiver.objective import TargetResponse, dose_response_match

SEEDS = 3
EVAL_BUDGET = 300
TUNE_STEPS = 2
COND_STEPS = 80
SPEC_STEPS = 40
BATCH = 12
BETA = 4.0
MAX_STEPS = 8

# "interpolation": hold out four in-grid corners (nearest-neighbor reuse is easy).
# "extrapolation": hold out the entire top EC50 decade (indices 8-11), so both the conditioned policy and
# the nearest-trained specialist must extrapolate beyond the trained EC50 range.
SPLIT = os.environ.get("SPLIT", "extrapolation")
HELD_OUT = {2, 5, 8, 11} if SPLIT == "interpolation" else {8, 9, 10, 11}
TRAIN_INDICES = [i for i in range(len(TARGETS)) if i not in HELD_OUT]

RESULTS = Path(__file__).parent / "results" / f"amortization_isolation_{SPLIT}.json"


def score_terminals(terminals: list[GRN], target: TargetResponse, seed: int) -> int:
    """Distinct high-scoring topologies among sampled terminals, scored against ``target``."""
    behavior = dose_response_match(target)
    rng = random.Random(seed)
    return distinct([(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS)) for g in terminals])


def nearest_trained(held_index: int) -> int:
    """The trained-target index closest to a held-out target in target-feature space."""
    hf = target_features(TARGETS[held_index])
    return min(
        TRAIN_INDICES,
        key=lambda j: sum((a - b) ** 2 for a, b in zip(hf, target_features(TARGETS[j]))),
    )


def main() -> None:
    n_samples = EVAL_BUDGET // (TUNE_STEPS + 1)
    print(
        f"SPLIT={SPLIT} | {SEEDS} seeds | {len(TARGETS)} targets "
        f"({len(HELD_OUT)} held out) | {n_samples} samples/target"
    )
    print(
        f"conditioned trained on {len(TRAIN_INDICES)} targets ({COND_STEPS * BATCH} sims); "
        f"specialists {SPEC_STEPS * BATCH} sims each\n"
    )

    # Aligned per-(seed, target) cells.
    tr_cond: list[int] = []
    tr_spec: list[int] = []
    ho_cond: list[int] = []
    ho_near: list[int] = []

    t0 = time.perf_counter()
    for seed in range(SEEDS):
        conditioned = train_conditioned(
            [TARGETS[i] for i in TRAIN_INDICES],
            steps=COND_STEPS,
            batch=BATCH,
            beta=BETA,
            max_steps=MAX_STEPS,
            seed=seed,
        )
        specialists: dict[int, PolicyNet] = {}
        for i in TRAIN_INDICES:
            specialists[i] = train(
                dose_response_match(TARGETS[i]),
                steps=SPEC_STEPS,
                batch=BATCH,
                beta=BETA,
                max_steps=MAX_STEPS,
                seed=seed,
            )

        # Trained targets: conditioned vs its own specialist.
        for i in TRAIN_INDICES:
            cond_terms = sample_conditioned(conditioned, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=3000 + seed)
            spec_terms = sample_terminals(specialists[i], n_samples, max_steps=MAX_STEPS, seed=3000 + seed)
            tr_cond.append(score_terminals(cond_terms, TARGETS[i], 3000 + seed))
            tr_spec.append(score_terminals(spec_terms, TARGETS[i], 3000 + seed))

        # Held-out targets: conditioned (zero-shot) vs nearest trained specialist (zero-shot).
        for i in sorted(HELD_OUT):
            donor = specialists[nearest_trained(i)]
            cond_terms = sample_conditioned(conditioned, TARGETS[i], n_samples, max_steps=MAX_STEPS, seed=4000 + seed)
            near_terms = sample_terminals(donor, n_samples, max_steps=MAX_STEPS, seed=4000 + seed)
            ho_cond.append(score_terminals(cond_terms, TARGETS[i], 4000 + seed))
            ho_near.append(score_terminals(near_terms, TARGETS[i], 4000 + seed))

        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    print(f"\nran {SEEDS} seeds in {time.perf_counter() - t0:.0f}s\n")

    print("== trained targets: does amortizing one policy cost anything vs per-target specialists? ==")
    print(
        f"  conditioned {mean(tr_cond):.1f} ± {pstdev(tr_cond):.1f} | "
        f"specialist {mean(tr_spec):.1f} ± {pstdev(tr_spec):.1f}"
    )
    trained = paired_summary(tr_cond, tr_spec)
    print(
        f"  paired conditioned − specialist: {trained.mean_diff:+.2f} "
        f"[95% CI {trained.ci_low:+.2f}, {trained.ci_high:+.2f}] "
        f"| win {trained.win_rate:.0%} | sign-test p={trained.sign_p:.1e} | n={trained.n}"
    )
    print("  (near zero => sharing is free; strongly negative => an amortization gap)\n")

    print("== held-out targets: conditioned vs nearest-trained-target zero-shot (the H1 test) ==")
    print(
        f"  conditioned {mean(ho_cond):.1f} ± {pstdev(ho_cond):.1f} | "
        f"nearest-reuse {mean(ho_near):.1f} ± {pstdev(ho_near):.1f}"
    )
    held = paired_summary(ho_cond, ho_near)
    print(
        f"  paired conditioned − nearest-reuse: {held.mean_diff:+.2f} [95% CI {held.ci_low:+.2f}, {held.ci_high:+.2f}] "
        f"| win {held.win_rate:.0%} | sign-test p={held.sign_p:.1e} | n={held.n}"
    )
    print("  (conditioned > nearest-reuse => amortization learned the target map, not just memorized anchors)")

    def _paired(s: PairedSummary) -> dict[str, float]:
        return {
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
            "eval_budget": EVAL_BUDGET,
            "tune_steps": TUNE_STEPS,
            "cond_sims": COND_STEPS * BATCH,
            "spec_sims": SPEC_STEPS * BATCH,
            "beta": BETA,
            "held_out": sorted(HELD_OUT),
        },
        "trained": {"conditioned_mean": mean(tr_cond), "specialist_mean": mean(tr_spec), "paired": _paired(trained)},
        "held_out": {"conditioned_mean": mean(ho_cond), "nearest_reuse_mean": mean(ho_near), "paired": _paired(held)},
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
