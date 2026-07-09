from __future__ import annotations

from quiver.eval import bootstrap_ci, paired_summary, random_effects, sign_test_p


def test_paired_summary_flags_a_consistent_advantage() -> None:
    # a beats b in every cell: full win rate, positive mean difference, CI above zero.
    a = [5, 3, 8, 6, 4]
    b = [2, 1, 5, 4, 2]
    s = paired_summary(a, b)
    assert s.n == 5
    assert s.win_rate == 1.0
    assert s.strict_win_rate == 1.0
    assert s.mean_diff > 0
    assert s.ci_low > 0
    assert s.sign_p < 0.1  # 5/5 wins is unlikely under the null


def test_paired_summary_sees_no_advantage_when_tied() -> None:
    a = [3, 3, 3, 3]
    s = paired_summary(a, a)
    assert s.mean_diff == 0.0
    assert s.strict_win_rate == 0.0
    assert s.win_rate == 1.0  # >= counts ties as (non-strict) wins
    assert s.sign_p == 1.0  # no non-tied cells


def test_paired_summary_pooled_std_would_hide_a_paired_win() -> None:
    # Targets differ wildly in difficulty (huge pooled spread) but a beats b within every cell.
    a = [1, 100, 2, 200]
    b = [0, 90, 1, 180]
    s = paired_summary(a, b)
    assert s.strict_win_rate == 1.0  # the paired signal is clean
    assert s.mean_diff > 0


def test_sign_test_symmetric_and_bounded() -> None:
    assert sign_test_p([1, 2, 3], [1, 2, 3]) == 1.0  # all ties
    assert 0.0 <= sign_test_p([2, 2, 2], [1, 1, 1]) <= 1.0
    # A single win vs a single loss is maximally inconclusive.
    assert sign_test_p([2, 0], [1, 1]) == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(values, seed=0, resamples=500)
    assert lo <= 3.0 <= hi
    assert bootstrap_ci(values, seed=0, resamples=500) == (lo, hi)  # seeded, reproducible
    assert bootstrap_ci([], seed=0) == (0.0, 0.0)


def test_random_effects_homogeneous_has_no_heterogeneity() -> None:
    # Identical effects with equal variance: pooled mean is the effect, tau2 and I^2 vanish.
    re = random_effects([0.5, 0.5, 0.5, 0.5], [0.01, 0.01, 0.01, 0.01])
    assert abs(re.mean - 0.5) < 1e-9
    assert re.tau2 == 0.0
    assert re.i2 == 0.0
    assert re.ci_low < 0.5 < re.ci_high


def test_random_effects_detects_heterogeneity_and_downweights_noisy_targets() -> None:
    # Spread-out effects with tight variances: heterogeneity is detected (tau2 > 0, high I^2).
    re = random_effects([-0.6, -0.4, -0.5, -0.2], [0.001, 0.001, 0.001, 0.001])
    assert re.tau2 > 0.0
    assert re.i2 > 0.5
    assert re.ci_low < re.mean < re.ci_high
    # A precise target pulls the pooled mean toward it more than a noisy one.
    precise = random_effects([-0.6, 0.0], [0.0001, 1.0])
    assert precise.mean < -0.4  # dominated by the low-variance −0.6


def test_random_effects_empty_and_singleton() -> None:
    assert random_effects([], []).k == 0
    one = random_effects([0.3], [0.04])
    assert one.k == 1 and abs(one.mean - 0.3) < 1e-9 and one.ci_low < 0.3 < one.ci_high
