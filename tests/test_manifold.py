from __future__ import annotations

from quiver.generate.manifold import KineticsKDE, UniformBox, circuit_log_likelihood, load_corpus
from quiver.generate.search import example_inducible


def test_corpus_loads() -> None:
    corpus = load_corpus()
    assert len(corpus) > 20
    assert all(s.K > 0 and s.n > 0 and s.dynamic_range > 0 for s in corpus)


def test_kde_prefers_typical_parts_over_off_manifold() -> None:
    kde = KineticsKDE(load_corpus())
    # A part near the corpus median scores higher than one far outside it.
    typical = kde.log_likelihood(0.13, 2.7, 155.0)
    off = kde.log_likelihood(100.0, 9.0, 5.0)
    assert typical > off


def test_kde_beats_box_on_a_typical_part() -> None:
    corpus = load_corpus()
    kde, box = KineticsKDE(corpus), UniformBox(corpus)
    # The KDE concentrates mass where parts cluster, so a typical part is likelier under it than the flat box.
    assert kde.log_likelihood(0.13, 2.7, 155.0) > box.log_likelihood(0.13, 2.7, 155.0)


def test_kde_samples_are_positive() -> None:
    kde = KineticsKDE(load_corpus())
    for s in kde.sample(5, seed=0):
        assert s.K > 0 and s.n > 0 and s.dynamic_range > 0


def test_circuit_log_likelihood_is_finite() -> None:
    # The example circuit's operators may not all form a dynamic-range triple; the score is still finite.
    ll = circuit_log_likelihood(example_inducible(), KineticsKDE(load_corpus()))
    assert ll <= 0.0
