"""Pooled headline estimates for the decorrelation results, derived from the per-target artifacts.

The scaled and cross-model decorrelation runs persist one record per target (failure-correlation and
survival gaps with their sampling variances). The paper reports the DerSimonian-Laird random-effects pooled
estimate over those records. This reads the committed per-target JSONs and writes the pooled headline
numbers (effect, 95% CI, share of targets in the effect's direction, tau^2, I^2, usable count) to a single
artifact, so every pooled figure in the decorrelation sections is a field in a file rather than a
re-derivation. It re-runs in under a second from the committed data, independent of the expensive scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from quiver.research.eval.stats import random_effects

RESULTS_DIR = Path(__file__).parent / "results"
OUT = RESULTS_DIR / "decorrelation_summary.json"


def _load(name: str) -> list[dict[str, float]]:
    return [r for r in json.loads((RESULTS_DIR / name).read_text()) if r["usable"]]


def _pool(records: list[dict[str, float]], effect: str, var: str) -> dict[str, float]:
    re = random_effects([r[effect] for r in records], [r[var] for r in records])
    # Share of targets strictly in the pooled effect's direction (zeros are not counted as in-direction).
    if re.mean < 0:
        share = sum(1 for r in records if r[effect] < 0) / len(records)
    else:
        share = sum(1 for r in records if r[effect] > 0) / len(records)
    return {
        "pooled": re.mean,
        "ci_low": re.ci_low,
        "ci_high": re.ci_high,
        "share_in_direction": share,
        "tau2": re.tau2,
        "i2": re.i2,
        "n_targets": re.k,
    }


def main() -> None:
    scaled = _load("decorrelation_scaled.json")
    failure_correlation = _pool(scaled, "corr_diff", "corr_var")
    parameter_survival = _pool(scaled, "surv_gap", "surv_var")
    summary: dict[str, object] = {
        "parameter_context": {
            "n_usable": len(scaled),
            "failure_correlation_distinct_minus_same": failure_correlation,
            "portfolio_survival_diverse_minus_clustered": parameter_survival,
            "mean_corr_same": fmean(r["corr_same"] for r in scaled),
            "mean_corr_distinct": fmean(r["corr_distinct"] for r in scaled),
            "mean_survival_diverse": fmean(r["surv_diverse"] for r in scaled),
            "mean_survival_clustered": fmean(r["surv_clustered"] for r in scaled),
        }
    }

    cross_headlines: list[tuple[str, int, dict[str, float], dict[str, float], float]] = []
    for label, name in (
        ("resource_competition", "decorrelation_crossmodel.json"),
        ("intrinsic_noise", "decorrelation_crossmodel_noise.json"),
    ):
        recs = _load(name)
        within = _pool(recs, "b_diff", "b_var")
        survival = _pool(recs, "surv_b_gap", "surv_b_var")
        fragility = fmean(r["cross_r"] for r in recs)
        summary[label] = {
            "n_usable": len(recs),
            "model_a_decorrelation": _pool(recs, "a_diff", "a_var"),
            "within_model_decorrelation": within,
            "portfolio_survival": survival,
            "mean_cross_model_fragility_r": fragility,
        }
        cross_headlines.append((label, len(recs), within, survival, fragility))

    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    def _ci(p: dict[str, float]) -> str:
        return f"{p['pooled']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] {p['share_in_direction']:.0%}"

    print(f"== parameter/context (model A), n = {len(scaled)} ==")
    print(f"  decorrelation {_ci(failure_correlation)}")
    print(f"  survival      {_ci(parameter_survival)}")
    for label, n_usable, within, survival, fragility in cross_headlines:
        print(f"== {label}, n = {n_usable} ==")
        print(
            f"  decorrelation {within['pooled']:+.3f} | " f"survival {survival['pooled']:+.3f} | r vs A {fragility:.2f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
