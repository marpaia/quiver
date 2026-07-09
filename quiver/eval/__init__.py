"""Evaluation: portfolio breadth (discovery curve), sampling quality, and a synthetic de-risk harness."""

from __future__ import annotations

from quiver.eval.decorrelation import (
    FailureCorrelation,
    FailureVector,
    failed_conditions,
    failure_correlation,
    marginal_pass_rate,
    portfolio_survival,
)
from quiver.eval.metrics import (
    PortfolioReport,
    QualityReport,
    discovery_curve,
    evaluate_portfolio,
    evaluate_quality,
    pass_at_k,
    sampling_fidelity,
)
from quiver.eval.select import select_portfolio
from quiver.eval.stats import PairedSummary, RandomEffects, bootstrap_ci, paired_summary, random_effects, sign_test_p
from quiver.eval.synthetic import concentrated_samples, proportional_samples

__all__ = [
    "FailureCorrelation",
    "FailureVector",
    "PairedSummary",
    "PortfolioReport",
    "QualityReport",
    "RandomEffects",
    "bootstrap_ci",
    "concentrated_samples",
    "discovery_curve",
    "evaluate_portfolio",
    "evaluate_quality",
    "failed_conditions",
    "failure_correlation",
    "marginal_pass_rate",
    "paired_summary",
    "pass_at_k",
    "portfolio_survival",
    "proportional_samples",
    "random_effects",
    "sampling_fidelity",
    "select_portfolio",
    "sign_test_p",
]
