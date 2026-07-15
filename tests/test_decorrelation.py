"""Failure-decorrelation metrics: correlation of failure vectors, portfolio survival, pass rate."""

from __future__ import annotations

from quiver.research.eval.decorrelation import (
    failed_conditions,
    failure_correlation,
    marginal_pass_rate,
    portfolio_survival,
)


def test_failed_conditions_thresholds_below() -> None:
    assert failed_conditions([0.6, 0.4, 0.5], 0.5) == (False, True, False)


def test_failure_correlation_splits_same_from_distinct() -> None:
    # Two identical variants of topology A (perfectly correlated failures), and topology B is the exact
    # complement of A (perfectly anti-correlated).
    a = (True, False, True, False)
    b = (False, True, False, True)
    result = failure_correlation([("A", a), ("A", a), ("B", b)])
    assert result.same == 1.0
    assert result.distinct == -1.0
    assert result.n_same == 1
    assert result.n_distinct == 2


def test_failure_correlation_excludes_constant_vectors() -> None:
    # An all-pass member has no variance, so every pair involving it is dropped (correlation undefined).
    varied = (True, False, True, False)
    constant = (False, False, False, False)
    result = failure_correlation([("A", varied), ("B", constant)])
    assert result.n_distinct == 0
    assert result.distinct is None


def test_portfolio_survival_is_fraction_of_conditions_with_a_survivor() -> None:
    # cond0: member1 passes; cond1: both fail; cond2: both pass -> 2 of 3 survived.
    vectors = [(True, True, False), (False, True, False)]
    assert portfolio_survival(vectors) == 2 / 3


def test_portfolio_survival_empty_is_zero() -> None:
    assert portfolio_survival([]) == 0.0


def test_marginal_pass_rate_counts_passes_over_cells() -> None:
    vectors = [(True, True, False), (False, True, False)]
    assert marginal_pass_rate(vectors) == 0.5
