"""Cross-model decorrelation: is the diversity hedge a property of structural distinctness, or an artifact
of one simulator's failure geometry?

Each design is scored under two mechanistically distinct held-out failure models: model A, the parameter and
assay-window context battery (:mod:`quiver.objective.stress`), and model B, shared-resource competition
(:mod:`quiver.objective.resource`), whose topology-dependent burden compresses achievable dynamic range.
Over a manifold of targets we report, pooled by DerSimonian--Laird random effects:

- within each model, the same- vs distinct-topology failure-correlation split (does decorrelation hold?);
- portfolio survival under model B, an unseen failure model the corpus was never tuned or selected against;
- cross-model disagreement: how weakly a design's model-A fragility predicts its model-B fragility, and how
  often a design that survives every model-A condition still fails under model B, evidence that B is a
  genuinely different geometry, so a hedge that holds under both is hedging model misspecification itself.

Run: ``uv run python experiments/decorrelation_crossmodel.py``; ``N_TARGETS=20 ...`` to sanity-check.
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
from typing import Callable

import _common

from quiver.eval import FailureVector, failed_conditions, failure_correlation, portfolio_survival, random_effects
from quiver.generate import evolutionary_search_designs, mutate, random_search_designs
from quiver.grn import GRN
from quiver.objective import (
    Behavior,
    TargetResponse,
    context_battery,
    dose_response_match,
    noise_battery,
    resource_battery,
    score_once,
    score_under_context,
    score_under_noise,
    score_under_resource,
    sensor_target_manifold,
)
from quiver.parallel import pmap

# Model B is selected by the MODEL env var: "resource" (shared-budget burden) or "noise" (intrinsic
# molecular noise, orthogonal to the deterministic models). Model A is always the parameter/context battery.
MODEL = os.environ.get("MODEL", "resource")
CONTEXT_BATTERY = context_battery()
RESOURCE_BATTERY = resource_battery()
NOISE_BATTERY = noise_battery(_common.env_int("NOISE_REPS", 12))
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

Select = Callable[["Design"], FailureVector]


@dataclass
class Design:
    topology: str
    grn: GRN
    nominal: float
    failures_a: FailureVector  # model A: parameter/context battery
    failures_b: FailureVector  # model B: resource-competition battery


@dataclass(frozen=True)
class TargetRecord:
    ec50: float
    fold: float
    direction: str
    n_topologies: int
    n_multi: int
    usable: bool
    reason: str
    a_same: float
    a_distinct: float
    a_diff: float
    a_var: float
    b_same: float
    b_distinct: float
    b_diff: float
    b_var: float
    surv_b_diverse: float
    surv_b_clustered: float
    surv_b_gap: float
    surv_b_var: float
    cross_r: float  # per-target Pearson(A-fail-rate, B-fail-rate) across designs
    b_catches: float  # fraction of designs passing all of A that fail >= 1 of B


def _fail_a(grn: GRN, behavior: Behavior) -> FailureVector:
    return failed_conditions([score_under_context(grn, behavior, c) for c in CONTEXT_BATTERY], STRESS)


def _fail_b(grn: GRN, behavior: Behavior) -> FailureVector:
    if MODEL == "noise":
        return failed_conditions([score_under_noise(grn, behavior, s) for s in NOISE_BATTERY], STRESS)
    return failed_conditions([score_under_resource(grn, behavior, c) for c in RESOURCE_BATTERY], STRESS)


def _make(grn: GRN, score: float, behavior: Behavior) -> Design:
    return Design(grn.topology_hash(), grn, score, _fail_a(grn, behavior), _fail_b(grn, behavior))


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


def _clustered_corr_var(by_topo: dict[str, list[Design]], select: Select) -> float:
    rng = random.Random(0)
    topos = list(by_topo)
    diffs: list[float] = []
    for _ in range(CLUSTER_RESAMPLES):
        draw = [rng.choice(topos) for _ in topos]
        items: list[tuple[str, FailureVector]] = []
        for k, h in enumerate(draw):
            for d in by_topo[h]:
                items.append((f"{h}#{k}", select(d)))
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
        take = min(len(by_topo[h]), k - len(picked))
        picked.extend(rng.sample(by_topo[h], take))
        if len(picked) >= k:
            return picked
    return None


def _survival_gap(by_topo: dict[str, list[Design]], select: Select) -> tuple[float, float, float, float] | None:
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
        d_surv = portfolio_survival([select(d) for d in diverse])
        c_surv = portfolio_survival([select(d) for d in clustered])
        div.append(d_surv)
        clu.append(c_surv)
        per_trial.append(d_surv - c_surv)
    if not per_trial:
        return None
    var = pvariance(per_trial) / len(per_trial) if len(per_trial) > 1 else 1e-9
    return (fmean(div), fmean(clu), fmean(per_trial), max(var, 1e-9))


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    vx: float = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy: float = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy)


def _excluded(target: TargetResponse, n_topo: int, n_multi: int, reason: str) -> TargetRecord:
    z = 0.0
    return TargetRecord(
        target.ec50,
        target.dynamic_range,
        target.direction,
        n_topo,
        n_multi,
        False,
        reason,
        z,
        z,
        z,
        1e-9,
        z,
        z,
        z,
        1e-9,
        z,
        z,
        z,
        1e-9,
        z,
        z,
    )


def analyze_target(target: TargetResponse) -> TargetRecord:
    behavior = dose_response_match(target)
    by_topo = _build_corpus(behavior)
    n_multi = sum(1 for v in by_topo.values() if len(v) >= 2)
    if len(by_topo) < K_PORTFOLIO:
        return _excluded(target, len(by_topo), n_multi, "too few distinct topologies")
    designs = [d for v in by_topo.values() for d in v]
    fc_a = failure_correlation([(d.topology, d.failures_a) for d in designs])
    fc_b = failure_correlation([(d.topology, d.failures_b) for d in designs])
    gap_b = _survival_gap(by_topo, lambda d: d.failures_b)
    if fc_a.same is None or fc_a.distinct is None or fc_b.same is None or fc_b.distinct is None or gap_b is None:
        return _excluded(target, len(by_topo), n_multi, "no computable failure split")
    a_rate = [fmean([1.0 if f else 0.0 for f in d.failures_a]) for d in designs]
    b_rate = [fmean([1.0 if f else 0.0 for f in d.failures_b]) for d in designs]
    robust_a = [d for d in designs if not any(d.failures_a)]
    b_catches = fmean([1.0 if any(d.failures_b) else 0.0 for d in robust_a]) if robust_a else 0.0
    return TargetRecord(
        target.ec50,
        target.dynamic_range,
        target.direction,
        len(by_topo),
        n_multi,
        True,
        "",
        fc_a.same,
        fc_a.distinct,
        fc_a.distinct - fc_a.same,
        _clustered_corr_var(by_topo, lambda d: d.failures_a),
        fc_b.same,
        fc_b.distinct,
        fc_b.distinct - fc_b.same,
        _clustered_corr_var(by_topo, lambda d: d.failures_b),
        gap_b[0],
        gap_b[1],
        gap_b[2],
        gap_b[3],
        _pearson(a_rate, b_rate),
        b_catches,
    )


def _forest(records: list[TargetRecord], effect: str, var: str, label: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    usable = sorted((r for r in records if r.usable), key=lambda r: getattr(r, effect))
    effects = [getattr(r, effect) for r in usable]
    errs = [1.96 * getattr(r, var) ** 0.5 for r in usable]
    pooled = random_effects(effects, [getattr(r, var) for r in usable])
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.axvspan(pooled.ci_low, pooled.ci_high, color="#e8590c", alpha=0.2, zorder=0)
    ax.axvline(pooled.mean, color="#e8590c", lw=1.2, zorder=1)
    ax.axvline(0.0, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.errorbar(
        effects,
        list(range(len(usable))),
        xerr=errs,
        fmt="o",
        ms=1.6,
        lw=0,
        elinewidth=0.3,
        capsize=0,
        color="#3b5bdb",
        alpha=0.5,
        zorder=2,
    )
    ax.margins(y=0.01)
    ax.set_yticks([])
    ax.set_ylabel(f"{pooled.k} targets (sorted)", fontsize=8)
    ax.set_xlabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"pooled {pooled.mean:+.3f} [{pooled.ci_low:+.3f}, {pooled.ci_high:+.3f}]  I2={pooled.i2:.0%}", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _pooled(records: list[TargetRecord], effect: str, var: str) -> str:
    usable = [r for r in records if r.usable]
    re = random_effects([getattr(r, effect) for r in usable], [getattr(r, var) for r in usable])
    return f"{re.mean:+.3f} [95% CI {re.ci_low:+.3f}, {re.ci_high:+.3f}]  I2={re.i2:.0%}"


def main() -> None:
    targets = sensor_target_manifold(_common.N_TARGETS, seed=MANIFOLD_SEED)
    b_battery = NOISE_BATTERY if MODEL == "noise" else RESOURCE_BATTERY
    print(
        f"cross-model decorrelation | model B = {MODEL} | "
        f"{_common.N_TARGETS} targets | max_regulators={_common.MAX_REGULATORS}",
        flush=True,
    )
    print(
        f"model A: {len(CONTEXT_BATTERY)} context conditions | model B ({MODEL}): {len(b_battery)} conditions",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stream_path = RESULTS_DIR / f"decorrelation_crossmodel_{MODEL}.jsonl"
    stream_path.write_text("")  # truncate; every target is appended here as it completes
    t0 = time.perf_counter()
    completed = [0]

    def _progress(index: int, rec: TargetRecord) -> None:
        completed[0] += 1
        with stream_path.open("a") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
        tag = f"EC50={rec.ec50:.2f} DR={rec.fold:.0f} {rec.direction}"
        detail = f"b_diff={rec.b_diff:+.2f} b_catches={rec.b_catches:.0%}" if rec.usable else f"excluded:{rec.reason}"
        print(f"  [{completed[0]}/{len(targets)}] {time.perf_counter()-t0:.0f}s  {tag}  {detail}", flush=True)

    records = pmap(analyze_target, targets, workers=_common.workers(), on_result=_progress)
    usable = [r for r in records if r.usable]
    print(f"\nanalyzed {len(records)} targets in {time.perf_counter()-t0:.0f}s | usable {len(usable)}", flush=True)
    for reason in sorted({r.reason for r in records if not r.usable}):
        print(f"  excluded ({reason}): {sum(1 for r in records if r.reason == reason)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"decorrelation_crossmodel_{MODEL}.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2)
    )
    with (RESULTS_DIR / f"decorrelation_crossmodel_{MODEL}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    if not usable:
        print("\nno usable targets")
        return

    a_neg = sum(r.a_diff < 0 for r in usable) / len(usable)
    b_neg = sum(r.b_diff < 0 for r in usable) / len(usable)
    print("\n== within-model decorrelation (distinct - same failure correlation; < 0 = decorrelates) ==")
    print(f"  model A (context):    pooled {_pooled(records, 'a_diff', 'a_var')} | share<0 {a_neg:.0%}")
    print(f"  model B ({MODEL}): pooled {_pooled(records, 'b_diff', 'b_var')} | share<0 {b_neg:.0%}")

    surv_pos = sum(r.surv_b_gap > 0 for r in usable) / len(usable)
    print("\n== portfolio survival under model B (diverse - clustered; > 0 = hedges the unseen model) ==")
    print(f"  pooled {_pooled(records, 'surv_b_gap', 'surv_b_var')} | share>0 {surv_pos:.0%}")

    cross = [r.cross_r for r in usable]
    catches = [r.b_catches for r in usable]
    print("\n== cross-model disagreement (B is a different geometry, not a re-run of A) ==")
    print(f"  per-design A-fragility vs B-fragility: mean Pearson r = {fmean(cross):+.2f} (low => models disagree)")
    print(f"  designs surviving ALL of A that still fail >=1 of B: {fmean(catches):.0%} (B catches failures A misses)")

    _forest(
        records, "b_diff", "b_var", f"{MODEL} failure corr. (distinct - same)", RESULTS_DIR / f"forest_{MODEL}_corr.pdf"
    )
    _forest(
        records,
        "surv_b_gap",
        "surv_b_var",
        f"{MODEL} survival (diverse - clustered)",
        RESULTS_DIR / f"forest_{MODEL}_surv.pdf",
    )
    print(f"\nforest plots -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
