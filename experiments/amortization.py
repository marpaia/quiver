"""Amortization: train one conditioned GFlowNet, design for many targets (including held-out).

A target-conditioned GFlowNet is trained *once* over a distribution of sensor targets, then sampled for
every target (including targets held out of training) and compared to per-target random and evolutionary
search. Per the pre-registration, the claim needs three things: breadth (GFN >= evolutionary on trained
targets), generalization (GFN holds up on *held-out* targets), and an amortization crossover (train-once
plus sample beats per-target search across K targets in total sims).

Run: ``uv run python experiments/amortization.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import random
import time
from statistics import mean, pstdev

from _common import SENSOR_TARGET_GRID as TARGETS
from _common import distinct

from quiver.eval import bootstrap_ci, paired_summary
from quiver.generate import evolutionary_search, random_search, tune_score
from quiver.learn.amortized import sample_conditioned, train_conditioned
from quiver.learn.gflownet import PolicyNet
from quiver.objective import TargetResponse, dose_response_match

SEEDS = 10
EVAL_BUDGET = 300
TUNE_STEPS = 2
TRAIN_STEPS = 80
TRAIN_BATCH = 12
BETA = 4.0
MAX_STEPS = 8

HELD_OUT = {2, 5, 8, 11}
TRAIN_TARGETS = [t for i, t in enumerate(TARGETS) if i not in HELD_OUT]


def gflownet_distinct(policy: PolicyNet, target: TargetResponse, seed: int) -> int:
    behavior = dose_response_match(target)
    rng = random.Random(2000 + seed)
    terminals = sample_conditioned(
        policy, target, EVAL_BUDGET // (TUNE_STEPS + 1), max_steps=MAX_STEPS, seed=2000 + seed
    )
    return distinct([(grn.topology_hash(), tune_score(grn, behavior, rng, TUNE_STEPS)) for grn in terminals])


def main() -> None:
    print(f"{SEEDS} seeds | {len(TARGETS)} targets ({len(HELD_OUT)} held out) | eval {EVAL_BUDGET} sims/target/method")
    print(f"GFlowNet trained once on {len(TRAIN_TARGETS)} targets ({TRAIN_STEPS * TRAIN_BATCH} sims, β={BETA})\n")

    methods = ("random", "evolutionary", "gflownet")
    # results[split][method] = list over (seed, target) of distinct high-scoring topologies, aligned across
    # methods by cell; cell_target[split] records the target index of each cell for the per-target view.
    results: dict[str, dict[str, list[int]]] = {split: {m: [] for m in methods} for split in ("train", "held-out")}
    cell_target: dict[str, list[int]] = {"train": [], "held-out": []}
    seed_totals: dict[str, list[int]] = {m: [] for m in methods}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        policy = train_conditioned(
            TRAIN_TARGETS, steps=TRAIN_STEPS, batch=TRAIN_BATCH, beta=BETA, max_steps=MAX_STEPS, seed=seed
        )
        totals = {m: 0 for m in methods}
        for i, target in enumerate(TARGETS):
            split = "held-out" if i in HELD_OUT else "train"
            behavior = dose_response_match(target)
            rnd = distinct(random_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed))
            evo = distinct(evolutionary_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed))
            gfn = gflownet_distinct(policy, target, seed)
            for method, count in (("random", rnd), ("evolutionary", evo), ("gflownet", gfn)):
                results[split][method].append(count)
                totals[method] += count
            cell_target[split].append(i)
        for method in methods:
            seed_totals[method].append(totals[method])
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    print("\ndistinct high-scoring topologies per target (pooled mean ± std over seeds × targets):")
    print("  (pooled std is dominated by between-target difficulty, not by which sampler wins, see paired)")
    for split in ("train", "held-out"):
        print(f"  {split}:")
        for method in methods:
            values = results[split][method]
            print(f"    {method:13s} {mean(values):.1f} ± {pstdev(values):.1f}")

    print("\n== paired advantage (per (seed, target) cell; cancels target difficulty) ==")
    for split in ("train", "held-out"):
        print(f"  {split}:")
        for opponent in ("evolutionary", "random"):
            s = paired_summary(results[split]["gflownet"], results[split][opponent])
            print(
                f"    gfn − {opponent:12s} +{s.mean_diff:.2f} [95% CI {s.ci_low:+.2f}, {s.ci_high:+.2f}] "
                f"| win {s.win_rate:.0%} (strict {s.strict_win_rate:.0%}) | sign-test p={s.sign_p:.1e} | n={s.n}"
            )

    print("\n== per-target mean over seeds (exposes difficulty spread) ==")
    for split in ("train", "held-out"):
        by_target: dict[int, dict[str, list[int]]] = {}
        for idx, ti in enumerate(cell_target[split]):
            by_target.setdefault(ti, {m: [] for m in methods})
            for method in methods:
                by_target[ti][method].append(results[split][method][idx])
        print(f"  {split} (target: gfn / evo / rnd):")
        for ti in sorted(by_target):
            t = TARGETS[ti]
            g, e, r = (mean(by_target[ti][m]) for m in methods)
            print(f"    EC50={t.ec50:<4} DR={t.dynamic_range:<5.0f} {t.direction:<4}: {g:.1f} / {e:.1f} / {r:.1f}")

    print("\n== decision rule ==")
    ho = results["held-out"]
    tr = results["train"]
    g_tr, e_tr = mean(tr["gflownet"]), mean(tr["evolutionary"])
    g_ho, e_ho, r_ho = mean(ho["gflownet"]), mean(ho["evolutionary"]), mean(ho["random"])
    verdict = "holds" if (g_ho >= e_ho and g_ho > r_ho) else "FAILS"
    amort = TRAIN_STEPS * TRAIN_BATCH / len(TARGETS)
    print(f"  breadth (train):           gfn {g_tr:.1f} vs evo {e_tr:.1f}")
    print(f"  generalization (held-out): gfn {g_ho:.1f} | evo {e_ho:.1f} | rnd {r_ho:.1f} -> {verdict}")
    print(f"  amortized training cost:   {amort:.0f} sims/target across {len(TARGETS)} targets")

    print("\n== efficiency (distinct designs per 1000 total simulations, mean over seeds ± 95% CI) ==")
    print(f"  (GFlowNet total sims include its one-time {TRAIN_STEPS * TRAIN_BATCH}-sim training)")
    sims = {
        "random": len(TARGETS) * EVAL_BUDGET,
        "evolutionary": len(TARGETS) * EVAL_BUDGET,
        "gflownet": TRAIN_STEPS * TRAIN_BATCH + len(TARGETS) * EVAL_BUDGET,
    }
    for method in ("random", "evolutionary", "gflownet"):
        per_k = [total / sims[method] * 1000 for total in seed_totals[method]]
        lo, hi = bootstrap_ci(per_k)
        print(
            f"  {method:13s} {mean(per_k):.2f} per 1000 sims [95% CI {lo:.2f}, {hi:.2f}]  "
            f"({mean(seed_totals[method]):.1f} designs / {sims[method]} sims)"
        )


if __name__ == "__main__":
    main()
