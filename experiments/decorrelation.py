"""Construct validity: does structural diversity decorrelate downstream failure?

The discovery metric counts distinct topology hashes clearing the pass bar. This asks whether that count
measures the risk a portfolio is meant to hedge. Two experiments, per ``docs/preregistration.md``:

* **A (mechanism).** Over a pooled corpus of passing designs (random + evolutionary, method-agnostic),
  compare pairwise failure correlation for parameter variants of one topology vs. distinct topologies, and
  the survival of quality-matched diverse vs. clustered portfolios. Matching on nominal score isolates
  structure from quality.
* **B (payoff).** Each method's discovered portfolio, run through both held-out checks, reporting
  P(≥ 1 member survives) and expected survivors alongside mean nominal score.

Held-out checks (never in the tuning reward): a shared context battery (global expression / affinity /
dose-window / growth shocks) and per-design parameter stress (a wider log-normal ensemble than tuning).

Run: ``uv run python experiments/decorrelation.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, mean

from quiver.eval import (
    FailureVector,
    bootstrap_ci,
    failed_conditions,
    failure_correlation,
    marginal_pass_rate,
    paired_summary,
    portfolio_survival,
)
from quiver.generate import evolutionary_search_designs, mutate, random_search_designs, tune_best
from quiver.grn import GRN
from quiver.learn.amortized import sample_conditioned, train_conditioned
from quiver.objective import (
    Behavior,
    TargetResponse,
    context_battery,
    dose_response_match,
    robustness,
    score_once,
    score_under_context,
)

BATTERY = context_battery()
PASS = 0.5  # nominal score to enter a portfolio / corpus
STRESS = 0.5  # survival bar under a single held-out condition
PARAM_JITTER = 0.5  # wider than the tuning jitter (0.4), so parameter stress is genuinely held out
PARAM_ENSEMBLE = 16
TUNE_STEPS = 2

# Experiment A
SEEDS_A = range(5)
CORPUS_BUDGET = 240
MAX_TOPOLOGIES = 20  # distinct passing topologies to carry into the corpus
VARIANTS_PER_TOPOLOGY = 6  # passing parameter variants generated per topology
VARIANT_TRIES = 30  # perturbation attempts allowed to find each variant
K_PORTFOLIO = 5
MATCH_TOL = 0.05
MATCH_TRIALS = 600

# Experiment B
SEEDS_B = range(3)
EVAL_BUDGET = 300
PARAM_BAR = 0.4  # a member survives parameter stress if its wide-ensemble robustness clears this
TRAIN_STEPS = 60
TRAIN_BATCH = 12
BETA = 4.0
MAX_STEPS = 8
TARGETS = [
    TargetResponse("GFP", ec50=e, dynamic_range=r, direction=d)
    for e in (0.1, 1.0, 10.0)
    for r in (10.0, 100.0)
    for d in ("up", "down")
]
HELD_OUT = {2, 5, 8, 11}
TRAIN_TARGETS = [t for i, t in enumerate(TARGETS) if i not in HELD_OUT]
TARGETS_B = [1, 4, 5, 11]  # two trained (1, 4), two held-out (5, 11)


@dataclass
class Design:
    """A discovered passing design and its behavior under the held-out checks."""

    topology: str
    grn: GRN
    nominal: float
    failures: FailureVector


def context_failures(grn: GRN, behavior: Behavior) -> FailureVector:
    return failed_conditions([score_under_context(grn, behavior, ctx) for ctx in BATTERY], STRESS)


def make_design(grn: GRN, score: float, behavior: Behavior) -> Design:
    return Design(grn.topology_hash(), grn, score, context_failures(grn, behavior))


def two_sample_bootstrap_ci(a: list[float], b: list[float], *, resamples: int = 2000) -> tuple[float, float]:
    """Percentile CI for the difference of means ``mean(a) - mean(b)`` of two independent samples."""
    if not a or not b:
        return (0.0, 0.0)
    rng = random.Random(0)
    diffs = sorted(
        fmean(a[rng.randrange(len(a))] for _ in a) - fmean(b[rng.randrange(len(b))] for _ in b)
        for _ in range(resamples)
    )
    return (diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))])


def passing_variants(base: GRN, behavior: Behavior, rng: random.Random, n: int) -> list[Design]:
    """Up to ``n`` passing parameter variants of ``base``'s topology, by perturbing parameters and keeping
    those that still clear the pass bar. ``mutate`` changes parameters only, so the topology is fixed."""
    out: list[Design] = []
    tries = 0
    while len(out) < n and tries < VARIANT_TRIES:
        tries += 1
        candidate = mutate(base, rng, jitter=0.4)
        score = score_once(candidate, behavior)
        if score >= PASS:
            out.append(make_design(candidate, score, behavior))
    return out


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def experiment_a() -> None:
    print("== Experiment A: mechanism (does distinct topology => decorrelated failure?) ==")
    target = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="up")
    behavior = dose_response_match(target)

    # Collect distinct passing topologies (best parameterization each) from a broad search.
    bases: dict[str, tuple[GRN, float]] = {}
    for seed in SEEDS_A:
        for designs in (
            random_search_designs(behavior, budget=CORPUS_BUDGET, tune_steps=TUNE_STEPS, seed=seed),
            evolutionary_search_designs(behavior, budget=CORPUS_BUDGET, tune_steps=TUNE_STEPS, seed=seed),
        ):
            for grn, score in designs:
                if score < PASS:
                    continue
                h = grn.topology_hash()
                if h not in bases or score > bases[h][1]:
                    bases[h] = (grn, score)

    # For each topology, generate a set of passing parameter variants, so same-topology pairs (the
    # correlated case) and clustered portfolios exist by construction.
    rng = random.Random(0)
    by_topo: dict[str, list[Design]] = defaultdict(list)
    for topology, (grn, score) in list(bases.items())[:MAX_TOPOLOGIES]:
        variants = [make_design(grn, score, behavior)]
        variants.extend(passing_variants(grn, behavior, rng, VARIANTS_PER_TOPOLOGY - 1))
        by_topo[topology] = variants
    corpus = [d for variants in by_topo.values() for d in variants]

    multi = {h: v for h, v in by_topo.items() if len(v) >= 2}
    print(
        f"  corpus: {len(corpus)} passing parameterizations over {len(by_topo)} topologies "
        f"({len(multi)} with ≥2 parameter variants)"
    )
    print(
        f"  battery marginal pass rate: {marginal_pass_rate([d.failures for d in corpus]):.2f} "
        f"(over {len(BATTERY)} shared conditions)"
    )

    fc = failure_correlation([(d.topology, d.failures) for d in corpus])
    print("\n  failure correlation (Pearson over the shared battery):")
    print(f"    same topology (parameter variants): {_fmt(fc.same)}  (n={fc.n_same} pairs)")
    print(f"    distinct topologies:                {_fmt(fc.distinct)}  (n={fc.n_distinct} pairs)")
    if fc.same is not None and fc.distinct is not None:
        lo, hi = two_sample_bootstrap_ci(list(fc.distinct_diffs), list(fc.same_diffs))
        print(
            f"    distinct − same: {(fc.distinct - fc.same):+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  "
            f"(decorrelation ⇔ CI below 0)"
        )

    # Quality-matched portfolio survival: diverse (K distinct topologies) vs clustered (few topologies).
    rng = random.Random(0)
    diffs: list[float] = []
    surv_div: list[float] = []
    surv_clu: list[float] = []
    pass_div: list[float] = []
    pass_clu: list[float] = []
    for _ in range(MATCH_TRIALS):
        diverse = _sample_diverse(rng, by_topo, K_PORTFOLIO)
        clustered = _sample_clustered(rng, by_topo, K_PORTFOLIO)
        if diverse is None or clustered is None:
            continue
        if abs(mean(d.nominal for d in diverse) - mean(d.nominal for d in clustered)) > MATCH_TOL:
            continue
        sd = portfolio_survival([d.failures for d in diverse])
        sc = portfolio_survival([d.failures for d in clustered])
        diffs.append(sd - sc)
        surv_div.append(sd)
        surv_clu.append(sc)
        pass_div.append(marginal_pass_rate([d.failures for d in diverse]))
        pass_clu.append(marginal_pass_rate([d.failures for d in clustered]))

    print(f"\n  quality-matched portfolio survival (size {K_PORTFOLIO}, {len(diffs)} matched pairs):")
    if diffs:
        clo, chi = bootstrap_ci(diffs)
        print(f"    diverse   P(≥1 survives) {mean(surv_div):.3f}  | marginal pass {mean(pass_div):.3f}")
        print(f"    clustered P(≥1 survives) {mean(surv_clu):.3f}  | marginal pass {mean(pass_clu):.3f}")
        print(f"    diverse − clustered: {mean(diffs):+.3f}  [95% CI {clo:+.3f}, {chi:+.3f}]  " f"(hedge ⇔ CI above 0)")
    else:
        print("    no matched pairs at this tolerance (insufficient clustered variants)")


def _sample_diverse(rng: random.Random, by_topo: dict[str, list[Design]], k: int) -> list[Design] | None:
    if len(by_topo) < k:
        return None
    return [rng.choice(by_topo[h]) for h in rng.sample(list(by_topo), k)]


def _sample_clustered(rng: random.Random, by_topo: dict[str, list[Design]], k: int) -> list[Design] | None:
    """A portfolio of ``k`` designs drawn from as few topologies as possible (parameter variants)."""
    order = sorted(by_topo, key=lambda h: len(by_topo[h]), reverse=True)
    picked: list[Design] = []
    for h in order:
        variants = by_topo[h]
        take = min(len(variants), k - len(picked))
        picked.extend(rng.sample(variants, take))
        if len(picked) >= k:
            return picked
    return None


def experiment_b() -> None:
    print("\n== Experiment B: per-method payoff (portfolio survival at the fixed budget) ==")
    methods = ("random", "evolutionary", "gflownet")
    ctx_surv: dict[str, list[float]] = {m: [] for m in methods}
    param_any: dict[str, list[float]] = {m: [] for m in methods}
    distinct: dict[str, list[int]] = {m: [] for m in methods}
    empty: dict[str, list[int]] = {m: [] for m in methods}
    member_nominal: dict[str, list[float]] = {m: [] for m in methods}
    member_robust: dict[str, list[float]] = {m: [] for m in methods}
    t0 = time.perf_counter()

    for seed in SEEDS_B:
        policy = train_conditioned(
            TRAIN_TARGETS, steps=TRAIN_STEPS, batch=TRAIN_BATCH, beta=BETA, max_steps=MAX_STEPS, seed=seed
        )
        for ti in TARGETS_B:
            target = TARGETS[ti]
            behavior = dose_response_match(target)
            rng = random.Random(3000 + seed)
            portfolios = {
                "random": _passers(
                    random_search_designs(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed), behavior
                ),
                "evolutionary": _passers(
                    evolutionary_search_designs(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed),
                    behavior,
                ),
                "gflownet": _passers(
                    [
                        tune_best(grn, behavior, rng, TUNE_STEPS)
                        for grn in sample_conditioned(
                            policy, target, EVAL_BUDGET // (TUNE_STEPS + 1), max_steps=MAX_STEPS, seed=3000 + seed
                        )
                    ],
                    behavior,
                ),
            }
            for method, recs in portfolios.items():
                distinct[method].append(len(recs))
                empty[method].append(0 if recs else 1)
                ctx_surv[method].append(portfolio_survival([d.failures for d in recs]) if recs else 0.0)
                robust = [
                    robustness(d.grn, behavior, ensemble=PARAM_ENSEMBLE, jitter=PARAM_JITTER, seed=seed) for d in recs
                ]
                member_robust[method].extend(robust)
                member_nominal[method].extend(d.nominal for d in recs)
                param_any[method].append(1.0 if any(r >= PARAM_BAR for r in robust) else 0.0)
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    def avg(xs: list[float]) -> float:
        return mean(xs) if xs else 0.0

    n_cells = len(list(SEEDS_B)) * len(TARGETS_B)
    print("\n  per method (mean over seed × target; nominal & param-robust pooled over portfolio members):")
    print(
        "    "
        + f"{'method':13s} {'distinct':>8s} {'empty':>6s} {'nominal':>8s} "
        + f"{'param-rob':>9s} {'ctxP≥1':>7s} {'paramP≥1':>9s}"
    )
    for method in methods:
        print(
            f"    {method:13s} {mean(distinct[method]):8.1f} {sum(empty[method]):6d} "
            f"{avg(member_nominal[method]):8.3f} {avg(member_robust[method]):9.3f} "
            f"{mean(ctx_surv[method]):7.3f} {mean(param_any[method]):9.3f}"
        )
    print(
        f"    (param-rob = mean member robustness at jitter {PARAM_JITTER}; paramP≥1 = P(≥1 member ≥ {PARAM_BAR}); "
        f"empty = portfolios with 0 passers of {n_cells} cells)"
    )

    print("\n  paired advantage on context survival (per seed × target cell):")
    for opponent in ("evolutionary", "random"):
        s = paired_summary(ctx_surv["gflownet"], ctx_surv[opponent])
        print(
            f"    gfn − {opponent:12s} {s.mean_diff:+.3f} [95% CI {s.ci_low:+.3f}, {s.ci_high:+.3f}] "
            f"| win {s.win_rate:.0%} | sign-test p={s.sign_p:.2g} | n={s.n}"
        )


def _passers(designs: list[tuple[GRN, float]], behavior: Behavior) -> list[Design]:
    """The distinct passing topologies, one best-nominal representative each, the delivered portfolio."""
    best: dict[str, tuple[GRN, float]] = {}
    for grn, score in designs:
        if score < PASS:
            continue
        h = grn.topology_hash()
        if h not in best or score > best[h][1]:
            best[h] = (grn, score)
    return [make_design(grn, score, behavior) for grn, score in best.values()]


def main() -> None:
    print(
        f"held-out checks: {len(BATTERY)} shared context conditions + parameter stress "
        f"(jitter {PARAM_JITTER}, ensemble {PARAM_ENSEMBLE}); survival bar {STRESS}\n"
    )
    experiment_a()
    experiment_b()


if __name__ == "__main__":
    main()
