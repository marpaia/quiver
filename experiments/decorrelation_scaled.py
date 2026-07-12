"""Scaled construct-validity test: does structural diversity decorrelate downstream failure, measured over
a large continuous manifold of sensor targets rather than a handful?

This is the hardened decorrelation experiment (``decorrelation_hardened.py``) scaled on two axes disclosed in
the pre-registration: the targets are drawn from a continuous manifold (``sensor_target_manifold``) at
``N_TARGETS`` rather than the discrete grid, and the corpora are built in the enlarged design space
(``max_regulators`` from the run config, default 12). The per-target effect keeps the topology-clustered
bootstrap of the hardened version; the headline statistic is now a **DerSimonian-Laird random-effects
interval** pooling the per-target effects by their precision, plus a forest plot of the per-target
distribution. Each target is analyzed independently, so the manifold is swept in parallel across targets.

Run: ``uv run python experiments/decorrelation_scaled.py``. Scale with the environment,
e.g. ``N_TARGETS=20 uv run python experiments/decorrelation_scaled.py`` for the sanity pass.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, mean, pvariance

import _common

from quiver.eval import (
    FailureVector,
    bootstrap_ci,
    failed_conditions,
    failure_correlation,
    portfolio_survival,
    random_effects,
)
from quiver.generate import evolutionary_search_designs, mutate, random_search_designs
from quiver.grn import GRN
from quiver.objective import (
    Behavior,
    TargetResponse,
    context_battery,
    dose_response_match,
    score_once,
    score_under_context,
    sensor_target_manifold,
)
from quiver.parallel import pmap

BATTERY = context_battery()
PASS = 0.5
STRESS = 0.5
MANIFOLD_SEED = 0
CORPUS_SEEDS = range(_common.env_int("CORPUS_SEEDS", 3))
CORPUS_BUDGET = 240
MAX_TOPOLOGIES = 12
VARIANTS_PER_TOPOLOGY = 6
VARIANT_TRIES = 30
TUNE_STEPS = 2
K_PORTFOLIO = 5
MATCH_TOL = 0.05
MATCH_TRIALS = 400
CLUSTER_RESAMPLES = 2000
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "experiments/results"))


@dataclass
class Design:
    topology: str
    grn: GRN
    nominal: float
    failures: FailureVector


@dataclass(frozen=True)
class TargetRecord:
    """Per-target decorrelation result. ``usable`` is false when the corpus is too small to compare or has
    no battery failures to hedge; ``reason`` says which. Effects carry their within-target sampling variance
    (``corr_var`` from the clustered bootstrap, ``surv_var`` from the matched-trial spread) so the pooled
    random-effects interval can weight targets by precision."""

    ec50: float
    fold: float
    direction: str
    n_topologies: int
    n_multi: int
    usable: bool
    reason: str
    corr_same: float
    corr_distinct: float
    corr_diff: float
    corr_var: float
    surv_diverse: float
    surv_clustered: float
    surv_gap: float
    surv_var: float


def _context_failures(grn: GRN, behavior: Behavior) -> FailureVector:
    return failed_conditions([score_under_context(grn, behavior, ctx) for ctx in BATTERY], STRESS)


def _make(grn: GRN, score: float, behavior: Behavior) -> Design:
    return Design(grn.topology_hash(), grn, score, _context_failures(grn, behavior))


def _passing_variants(base: GRN, behavior: Behavior, rng: random.Random, n: int) -> list[Design]:
    out: list[Design] = []
    tries = 0
    while len(out) < n and tries < VARIANT_TRIES:
        tries += 1
        candidate = mutate(base, rng, jitter=0.4)
        score = score_once(candidate, behavior)
        if score >= PASS:
            out.append(_make(candidate, score, behavior))
    return out


def _build_corpus(behavior: Behavior) -> dict[str, list[Design]]:
    bases: dict[str, tuple[GRN, float]] = {}
    for seed in CORPUS_SEEDS:
        for designs in (
            random_search_designs(
                behavior, budget=CORPUS_BUDGET, tune_steps=TUNE_STEPS, seed=seed, max_regulators=_common.MAX_REGULATORS
            ),
            evolutionary_search_designs(
                behavior, budget=CORPUS_BUDGET, tune_steps=TUNE_STEPS, seed=seed, max_regulators=_common.MAX_REGULATORS
            ),
        ):
            for grn, score in designs:
                if score < PASS:
                    continue
                h = grn.topology_hash()
                if h not in bases or score > bases[h][1]:
                    bases[h] = (grn, score)
    rng = random.Random(0)
    by_topo: dict[str, list[Design]] = {}
    for topology, (grn, score) in list(bases.items())[:MAX_TOPOLOGIES]:
        variants = [_make(grn, score, behavior)]
        variants.extend(_passing_variants(grn, behavior, rng, VARIANTS_PER_TOPOLOGY - 1))
        by_topo[topology] = variants
    return by_topo


def _clustered_corr_var(by_topo: dict[str, list[Design]]) -> float:
    """Sampling variance of the (distinct − same) failure-correlation estimate, from a topology-clustered
    bootstrap: resample topologies (each with its variant set) with replacement, recompute the split."""
    rng = random.Random(0)
    topos = list(by_topo)
    diffs: list[float] = []
    for _ in range(CLUSTER_RESAMPLES):
        draw = [rng.choice(topos) for _ in topos]
        items: list[tuple[str, FailureVector]] = []
        for k, h in enumerate(draw):
            for d in by_topo[h]:
                items.append((f"{h}#{k}", d.failures))
        fc = failure_correlation(items)
        if fc.same is not None and fc.distinct is not None:
            diffs.append(fc.distinct - fc.same)
    return pvariance(diffs) if len(diffs) > 1 else 1e-9


def _sample_diverse(rng: random.Random, by_topo: dict[str, list[Design]], k: int) -> list[Design] | None:
    if len(by_topo) < k:
        return None
    return [rng.choice(by_topo[h]) for h in rng.sample(list(by_topo), k)]


def _sample_clustered(rng: random.Random, by_topo: dict[str, list[Design]], k: int) -> list[Design] | None:
    order = sorted(by_topo, key=lambda h: len(by_topo[h]), reverse=True)
    picked: list[Design] = []
    for h in order:
        variants = by_topo[h]
        take = min(len(variants), k - len(picked))
        picked.extend(rng.sample(variants, take))
        if len(picked) >= k:
            return picked
    return None


def _survival_gap(by_topo: dict[str, list[Design]]) -> tuple[float, float, float, float] | None:
    """(diverse survival, clustered survival, gap, gap sampling variance) at matched nominal quality."""
    rng = random.Random(0)
    div: list[float] = []
    clu: list[float] = []
    per_trial: list[float] = []
    for _ in range(MATCH_TRIALS):
        diverse = _sample_diverse(rng, by_topo, K_PORTFOLIO)
        clustered = _sample_clustered(rng, by_topo, K_PORTFOLIO)
        if diverse is None or clustered is None:
            continue
        if abs(mean(d.nominal for d in diverse) - mean(d.nominal for d in clustered)) > MATCH_TOL:
            continue
        d_surv = portfolio_survival([d.failures for d in diverse])
        c_surv = portfolio_survival([d.failures for d in clustered])
        div.append(d_surv)
        clu.append(c_surv)
        per_trial.append(d_surv - c_surv)
    if not per_trial:
        return None
    var = pvariance(per_trial) / len(per_trial) if len(per_trial) > 1 else 1e-9
    return (fmean(div), fmean(clu), fmean(per_trial), max(var, 1e-9))


def _excluded(target: TargetResponse, n_topo: int, n_multi: int, reason: str) -> TargetRecord:
    return TargetRecord(
        target.ec50,
        target.dynamic_range,
        target.direction,
        n_topo,
        n_multi,
        False,
        reason,
        0.0,
        0.0,
        0.0,
        1e-9,
        0.0,
        0.0,
        0.0,
        1e-9,
    )


def analyze_target(target: TargetResponse) -> TargetRecord:
    """Full per-target decorrelation analysis, run in a worker. Serial in-process (no nested pool)."""
    behavior = dose_response_match(target)
    by_topo = _build_corpus(behavior)
    n_multi = sum(1 for v in by_topo.values() if len(v) >= 2)
    if len(by_topo) < K_PORTFOLIO:
        return _excluded(target, len(by_topo), n_multi, "too few distinct topologies")
    corpus = [(d.topology, d.failures) for v in by_topo.values() for d in v]
    fc = failure_correlation(corpus)
    if fc.same is None or fc.distinct is None:
        return _excluded(target, len(by_topo), n_multi, "no battery failures to correlate")
    gap = _survival_gap(by_topo)
    if gap is None:
        return _excluded(target, len(by_topo), n_multi, "no quality-matched portfolios")
    return TargetRecord(
        target.ec50,
        target.dynamic_range,
        target.direction,
        len(by_topo),
        n_multi,
        True,
        "",
        fc.same,
        fc.distinct,
        fc.distinct - fc.same,
        _clustered_corr_var(by_topo),
        gap[0],
        gap[1],
        gap[2],
        gap[3],
    )


def _write_forest(records: list[TargetRecord], effect: str, var: str, label: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable, skipping forest plot)")
        return
    usable = sorted((r for r in records if r.usable), key=lambda r: getattr(r, effect))
    effects = [getattr(r, effect) for r in usable]
    errs = [1.96 * getattr(r, var) ** 0.5 for r in usable]
    pooled = random_effects(effects, [getattr(r, var) for r in usable])
    # Fixed compact size (a caterpillar plot): the row count sets marker density, not figure height, so the
    # figure fits a paper column regardless of how many targets there are.
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ys = list(range(len(usable)))
    ax.axvspan(pooled.ci_low, pooled.ci_high, color="#e8590c", alpha=0.2, zorder=0)
    ax.axvline(pooled.mean, color="#e8590c", lw=1.2, zorder=1)
    ax.axvline(0.0, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.errorbar(
        effects, ys, xerr=errs, fmt="o", ms=1.6, lw=0, elinewidth=0.3, capsize=0, color="#3b5bdb", alpha=0.5, zorder=2
    )
    ax.margins(y=0.01)
    ax.set_yticks([])
    ax.set_ylabel(f"{pooled.k} targets (sorted)", fontsize=8)
    ax.set_xlabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"pooled {pooled.mean:+.3f} [{pooled.ci_low:+.3f}, {pooled.ci_high:+.3f}]  I2={pooled.i2:.0%}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  forest plot -> {path}")


def main() -> None:
    n_targets = _common.N_TARGETS
    targets = sensor_target_manifold(n_targets, seed=MANIFOLD_SEED)
    print(
        f"scaled decorrelation | {n_targets} manifold targets (seed {MANIFOLD_SEED}) "
        f"| max_regulators={_common.MAX_REGULATORS}"
    )
    print(
        f"corpus per target: random+evolutionary over {list(CORPUS_SEEDS)} seeds, "
        f"budget {CORPUS_BUDGET} | battery {len(BATTERY)} conditions"
    )

    t0 = time.perf_counter()
    records = pmap(analyze_target, targets, workers=_common.workers())
    dt = time.perf_counter() - t0

    usable = [r for r in records if r.usable]
    excluded = [r for r in records if not r.usable]
    print(f"\nanalyzed {len(records)} targets in {dt:.0f}s | usable {len(usable)} | excluded {len(excluded)}")
    for reason in sorted({r.reason for r in excluded}):
        print(f"  excluded ({reason}): {sum(1 for r in excluded if r.reason == reason)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "decorrelation_scaled.json").write_text(json.dumps([asdict(r) for r in records], indent=2))
    with (RESULTS_DIR / "decorrelation_scaled.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    if not usable:
        print("\nno usable targets, nothing to pool")
        return

    corr_effects = [r.corr_diff for r in usable]
    surv_effects = [r.surv_gap for r in usable]
    corr_re = random_effects(corr_effects, [r.corr_var for r in usable])
    surv_re = random_effects(surv_effects, [r.surv_var for r in usable])
    corr_boot = bootstrap_ci(corr_effects)
    surv_boot = bootstrap_ci(surv_effects)

    corr_median = sorted(corr_effects)[len(corr_effects) // 2]
    corr_share = sum(c < 0 for c in corr_effects) / len(corr_effects)
    print("\n== failure-correlation (distinct - same); decorrelation <=> below 0 ==")
    print(f"  per-target mean {mean(corr_effects):+.3f} | median {corr_median:+.3f} | share<0 {corr_share:.0%}")
    print(
        f"  random-effects pooled {corr_re.mean:+.3f} [95% CI {corr_re.ci_low:+.3f}, {corr_re.ci_high:+.3f}]  "
        f"tau2={corr_re.tau2:.3f}  I2={corr_re.i2:.0%}"
    )
    print(f"  bootstrap over targets  [{corr_boot[0]:+.3f}, {corr_boot[1]:+.3f}]")

    surv_median = sorted(surv_effects)[len(surv_effects) // 2]
    surv_share = sum(s > 0 for s in surv_effects) / len(surv_effects)
    print("\n== portfolio survival (diverse - clustered); hedge <=> above 0 ==")
    print(f"  per-target mean {mean(surv_effects):+.3f} | median {surv_median:+.3f} | share>0 {surv_share:.0%}")
    print(
        f"  random-effects pooled {surv_re.mean:+.3f} [95% CI {surv_re.ci_low:+.3f}, {surv_re.ci_high:+.3f}]  "
        f"tau2={surv_re.tau2:.3f}  I2={surv_re.i2:.0%}"
    )
    print(f"  bootstrap over targets  [{surv_boot[0]:+.3f}, {surv_boot[1]:+.3f}]")

    _write_forest(
        records,
        "corr_diff",
        "corr_var",
        "failure correlation (distinct - same)",
        RESULTS_DIR / "forest_correlation.pdf",
    )
    _write_forest(
        records, "surv_gap", "surv_var", "portfolio survival (diverse - clustered)", RESULTS_DIR / "forest_survival.pdf"
    )


if __name__ == "__main__":
    main()
