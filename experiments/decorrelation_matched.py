"""Base-rate-matched survival gap (review item R3): does the diverse-portfolio survival advantage survive
matching diverse and clustered portfolios on their held-out marginal pass rate, not only on nominal score?

The scaled decorrelation run matches portfolios on mean nominal detector score. As its own limitations note,
that leaves the survival gap able to fold in a small per-member robustness difference: a diverse set could
survive more often simply by having members that pass the held-out battery more often, independent of whether
their failures decorrelate. This run adds the missing control. It reuses the corpus construction and
portfolio sampling of ``decorrelation_scaled`` and recomputes the survival gap under the stricter constraint
that the diverse and clustered portfolios also match on ``marginal_pass_rate`` (the mean per-member,
per-condition held-out pass rate). The residual gap is then attributable to decorrelation rather than base
rate.

Run: ``uv run python experiments/decorrelation_matched.py`` (``sim`` extra). Shares the scale knobs of
``decorrelation_scaled`` (``N_TARGETS``, ``MAX_REGULATORS``, ``CORPUS_SEEDS``); ``PR_TOL`` sets the pass-rate
match tolerance (default 0.05).
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from statistics import fmean, mean, pvariance

import _common
import decorrelation_scaled as ds

from quiver.eval import bootstrap_ci, marginal_pass_rate, portfolio_survival, random_effects
from quiver.objective import TargetResponse, dose_response_match, sensor_target_manifold
from quiver.parallel import pmap

PR_TOL = _common.env_float("PR_TOL", 0.05)
RESULTS_DIR = ds.RESULTS_DIR


@dataclass(frozen=True)
class MatchedRecord:
    ec50: float
    fold: float
    direction: str
    n_topologies: int
    usable: bool
    reason: str
    n_matched_trials: int
    surv_diverse: float
    surv_clustered: float
    surv_gap: float
    surv_var: float


def _record(
    target: TargetResponse,
    n_topo: int,
    usable: bool,
    reason: str,
    n_trials: int,
    sd: float,
    sc: float,
    gap: float,
    var: float,
) -> MatchedRecord:
    return MatchedRecord(
        target.ec50, target.dynamic_range, target.direction, n_topo, usable, reason, n_trials, sd, sc, gap, var
    )


def _matched_survival_gap(by_topo: dict[str, list[ds.Design]]) -> tuple[float, float, float, float, int] | None:
    """(diverse survival, clustered survival, gap, gap sampling variance, matched-trial count) over trials
    where the two portfolios match on BOTH mean nominal score and marginal held-out pass rate."""
    rng = random.Random(0)
    div: list[float] = []
    clu: list[float] = []
    per_trial: list[float] = []
    for _ in range(ds.MATCH_TRIALS):
        diverse = ds._sample_diverse(rng, by_topo, ds.K_PORTFOLIO)
        clustered = ds._sample_clustered(rng, by_topo, ds.K_PORTFOLIO)
        if diverse is None or clustered is None:
            continue
        if abs(mean(d.nominal for d in diverse) - mean(d.nominal for d in clustered)) > ds.MATCH_TOL:
            continue
        d_fail = [d.failures for d in diverse]
        c_fail = [d.failures for d in clustered]
        if abs(marginal_pass_rate(d_fail) - marginal_pass_rate(c_fail)) > PR_TOL:
            continue
        d_surv = portfolio_survival(d_fail)
        c_surv = portfolio_survival(c_fail)
        div.append(d_surv)
        clu.append(c_surv)
        per_trial.append(d_surv - c_surv)
    if not per_trial:
        return None
    var = pvariance(per_trial) / len(per_trial) if len(per_trial) > 1 else 1e-9
    return (fmean(div), fmean(clu), fmean(per_trial), max(var, 1e-9), len(per_trial))


def analyze_target(target: TargetResponse) -> MatchedRecord:
    behavior = dose_response_match(target)
    by_topo = ds._build_corpus(behavior)
    if len(by_topo) < ds.K_PORTFOLIO:
        return _record(target, len(by_topo), False, "too few distinct topologies", 0, 0.0, 0.0, 0.0, 1e-9)
    gap = _matched_survival_gap(by_topo)
    if gap is None:
        return _record(target, len(by_topo), False, "no pass-rate-matched portfolios", 0, 0.0, 0.0, 0.0, 1e-9)
    return _record(target, len(by_topo), True, "", gap[4], gap[0], gap[1], gap[2], gap[3])


def main() -> None:
    n_targets = _common.N_TARGETS
    targets = sensor_target_manifold(n_targets, seed=ds.MANIFOLD_SEED)
    print(
        f"base-rate-matched survival | {n_targets} manifold targets (seed {ds.MANIFOLD_SEED}) "
        f"| PR_TOL={PR_TOL} | max_regulators={_common.MAX_REGULATORS}"
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
    (RESULTS_DIR / "decorrelation_matched.json").write_text(json.dumps([asdict(r) for r in records], indent=2))
    with (RESULTS_DIR / "decorrelation_matched.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    if not usable:
        print("\nno usable (pass-rate-matched) targets, nothing to pool")
        return

    surv_effects = [r.surv_gap for r in usable]
    surv_re = random_effects(surv_effects, [r.surv_var for r in usable])
    surv_boot = bootstrap_ci(surv_effects)
    share = sum(s > 0 for s in surv_effects) / len(surv_effects)
    trials = [r.n_matched_trials for r in usable]
    print("\n== base-rate-matched portfolio survival (diverse - clustered); hedge <=> above 0 ==")
    print(f"  matched trials/target: min {min(trials)}, median {sorted(trials)[len(trials) // 2]}, max {max(trials)}")
    print(f"  per-target mean {mean(surv_effects):+.3f} | share>0 {share:.0%}")
    print(
        f"  random-effects pooled {surv_re.mean:+.3f} [95% CI {surv_re.ci_low:+.3f}, {surv_re.ci_high:+.3f}]  "
        f"tau2={surv_re.tau2:.3f}  I2={surv_re.i2:.0%}"
    )
    print(f"  bootstrap over targets  [{surv_boot[0]:+.3f}, {surv_boot[1]:+.3f}]")
    print("\ncompare against the nominal-only-matched pooled gap from decorrelation_scaled.py (+0.20).")


if __name__ == "__main__":
    main()
