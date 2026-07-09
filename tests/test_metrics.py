from __future__ import annotations

from math import comb

from quiver.eval import discovery_curve, evaluate_portfolio, evaluate_quality, pass_at_k, sampling_fidelity


def test_sampling_fidelity_entropy_top_and_tv() -> None:
    # Topology "a" emitted 8×, "b" 2×, all passing with equal reward 0.6: entropy of (0.8, 0.2), top-mode
    # 0.8, and TV 0.3 to the (uniform, since rewards are equal) reward-proportional target at beta=1.
    hashes = ["a"] * 8 + ["b"] * 2
    scores = [0.6] * 10
    entropy, top_mode, tv = sampling_fidelity(hashes, scores, threshold=0.5, beta=1.0)
    assert top_mode is not None and abs(top_mode - 0.8) < 1e-9
    assert entropy is not None and abs(entropy - 0.7219) < 1e-3
    assert tv is not None and abs(tv - 0.3) < 1e-9


def test_sampling_fidelity_filters_and_degenerate() -> None:
    # Below-threshold samples are dropped; a single surviving topology is degenerate -> all None.
    entropy, top_mode, tv = sampling_fidelity(["a", "a", "b"], [0.6, 0.6, 0.2], threshold=0.5, beta=1.0)
    assert (entropy, top_mode, tv) == (None, None, None)


def test_pass_at_k_bounds() -> None:
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 0) == 0.0


def test_pass_at_k_formula() -> None:
    assert abs(pass_at_k(8, 3, 2) - (1.0 - comb(5, 2) / comb(8, 2))) < 1e-9


def test_discovery_curve_counts_cumulative_distinct_valid_modes() -> None:
    curve = discovery_curve(["a", "a", "b", None, "c"], passed=[True, True, True, False, False])
    assert curve == (1, 1, 2, 2, 2)


def test_discovery_curve_empty() -> None:
    assert discovery_curve([], passed=[]) == ()


def test_evaluate_portfolio_reports_breadth_and_curve() -> None:
    report = evaluate_portfolio(
        [1.0, 1.0, 1.0, 1.0],
        ["a", "b", "c", "d"],
        passed=[True, True, True, True],
        total_modes=8,
    )
    assert report.passers == 4 and report.distinct_passers == 4
    assert abs(report.diversity() - 1.0) < 1e-9
    assert abs(report.coverage() - 0.5) < 1e-9
    assert report.discovery == (1, 2, 3, 4)
    assert report.modes_at(0.5) == 2 and report.modes_at(1.0) == 4


def test_evaluate_quality_proportional_sampling() -> None:
    q = evaluate_quality(
        ["a", "a", "a", "a", "b"],
        [0.8, 0.8, 0.8, 0.8, 0.2],
        passed=[True, True, True, True, True],
    )
    assert q.n_passers == 5 and q.modes_hit == 2
    assert abs(q.mean_score - (0.8 * 4 + 0.2) / 5) < 1e-9
    assert q.score_tv is not None and q.score_tv < 1e-9
    assert q.top_mode_share is not None and abs(q.top_mode_share - 0.8) < 1e-9


def test_evaluate_quality_beta_targets_tempered_reward() -> None:
    # Mode a (score 0.8) is sampled 4×, b (0.2) once — proportional to score, so TV to the score^1 target
    # is ~0. Against a tempered score^2 target the same sampling is no longer proportional, so TV rises.
    hashes = ["a", "a", "a", "a", "b"]
    scores = [0.8, 0.8, 0.8, 0.8, 0.2]
    passed = [True, True, True, True, True]
    proportional = evaluate_quality(hashes, scores, passed=passed, beta=1.0)
    tempered = evaluate_quality(hashes, scores, passed=passed, beta=2.0)
    assert proportional.score_tv is not None and proportional.score_tv < 1e-9
    assert tempered.score_tv is not None and tempered.score_tv > 0.1


def test_evaluate_quality_mode_entropy_and_empty() -> None:
    uniform = evaluate_quality(["a", "b"], [0.5, 0.5], passed=[True, True])
    assert uniform.mode_entropy is not None and abs(uniform.mode_entropy - 1.0) < 1e-9
    single = evaluate_quality(["a", "a"], [0.5, 0.5], passed=[True, True])
    assert single.mode_entropy is None
    empty = evaluate_quality([], [], passed=[])
    assert empty.n_passers == 0 and empty.score_tv is None
