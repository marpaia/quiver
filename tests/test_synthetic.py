"""The metric instrument, validated against sampling distributions with a known answer — so a
measurement bug cannot masquerade as a result. Split into the two regimes where score_tv and the
discovery curve are each meaningful (proportionality needs N >> #modes; discovery rate needs N <~ #modes).
"""

from __future__ import annotations

from quiver.eval import discovery_curve, evaluate_portfolio, evaluate_quality
from quiver.eval.synthetic import concentrated_samples, proportional_samples


def test_metric_recovers_proportionality_contrast() -> None:
    scores = {"m0": 0.9, "m1": 0.7, "m2": 0.5, "m3": 0.3, "m4": 0.1}
    n = 6000
    prop = proportional_samples(scores, n, seed=1234)
    conc = concentrated_samples(scores, n, seed=1234)
    ok = [True] * n

    pq = evaluate_quality(prop, [scores[m] for m in prop], passed=ok)
    cq = evaluate_quality(conc, [scores[m] for m in conc], passed=ok)

    assert pq.score_tv is not None and pq.score_tv < 0.05
    assert pq.modes_hit == 5
    assert pq.mode_entropy is not None and pq.mode_entropy > 0.85
    assert pq.top_mode_share is not None and 0.30 < pq.top_mode_share < 0.42

    assert cq.score_tv is not None and cq.score_tv > 0.4
    assert cq.top_mode_share is not None and cq.top_mode_share > 0.85
    assert cq.mode_entropy is not None and cq.mode_entropy < 0.4

    assert cq.score_tv > pq.score_tv and pq.mode_entropy > cq.mode_entropy


def test_discovery_curve_separates_portfolio_breadth() -> None:
    n_modes, n = 200, 800
    scores = {f"m{i}": 1.0 for i in range(n_modes)}  # flat: proportional sampling is uniform, maximizing breadth
    ok = [True] * n

    prop = proportional_samples(scores, n, seed=7)
    conc = concentrated_samples(scores, n, seed=7, mass=0.9)

    assert discovery_curve(prop, passed=ok)[-1] > 150
    assert discovery_curve(prop, passed=ok)[-1] > discovery_curve(conc, passed=ok)[-1] + 40

    prop_q = evaluate_portfolio([1.0] * n, prop, passed=ok).modes_at(0.25)
    conc_q = evaluate_portfolio([1.0] * n, conc, passed=ok).modes_at(0.25)
    assert prop_q is not None and conc_q is not None and prop_q > 2 * conc_q
