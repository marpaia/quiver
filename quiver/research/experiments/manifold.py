"""W3 go/no-go: is a learned realizable-kinetics manifold worth claiming over hand-set ranges?

The plan's decision rule (condition i): the learned density is worth claiming iff, on held-out parts, its
predictive likelihood beats the hand-set ranges with a confidence interval excluding zero. This runs that
test on the Cello corpus, two ways, because they answer different questions:

- **Leave-one-out (within-corpus):** for each part, fit both models on the other parts and score the held-out
  part. Does the density generalize to unseen parts *of the families it has seen*? If the KDE wins, the
  density captures real structure (concentration, correlation) a flat box misses.
- **Leave-one-family-out (cross-family):** fit on all-but-one family, score the held-out family. Does the
  density generalize to a family it has *never* seen? This is the honest test of whether a manifold learned
  from a thin corpus (here two gate families) transfers, or is family-specific and data-gated.

A learned manifold that wins leave-one-out but loses leave-one-family-out is real but not yet general: the
verdict is "catalog-grounded ranges suffice until the corpus spans more families," which motivates growing
it (the flapjack-data Hill-fit layer, Marionette, public Flapjack studies).

Run: ``uv run python experiments/manifold.py``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from quiver.research.eval import bootstrap_ci
from quiver.research.manifold import DensityModel, KineticsKDE, KineticsSample, UniformBox, load_corpus

RESULTS = Path(__file__).parent / "results" / "manifold.json"


def _ll(model: DensityModel, sample: KineticsSample) -> float:
    return model.log_likelihood(sample.K, sample.n, sample.dynamic_range)


def leave_one_out(corpus: list[KineticsSample]) -> dict[str, float]:
    """Per-part held-out log-likelihood, KDE minus box, over the whole corpus."""
    diffs: list[float] = []
    kde_lls: list[float] = []
    box_lls: list[float] = []
    for i in range(len(corpus)):
        train = corpus[:i] + corpus[i + 1 :]
        held = corpus[i]
        kde, box = KineticsKDE(train), UniformBox(train)
        k, b = _ll(kde, held), _ll(box, held)
        kde_lls.append(k)
        box_lls.append(b)
        diffs.append(k - b)
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(diffs),
        "kde_mean_ll": mean(kde_lls),
        "box_mean_ll": mean(box_lls),
        "kde_minus_box_mean": mean(diffs),
        "ci_low": lo,
        "ci_high": hi,
        "kde_wins_frac": mean(1.0 if d > 0 else 0.0 for d in diffs),
    }


def leave_one_family_out(corpus: list[KineticsSample]) -> dict[str, dict[str, float]]:
    """Hold out each family entirely; fit on the rest and score the held-out family."""
    by_family: dict[str, list[KineticsSample]] = defaultdict(list)
    for s in corpus:
        by_family[s.family].append(s)
    folds: dict[str, dict[str, float]] = {}
    for family, held in by_family.items():
        train = [s for s in corpus if s.family != family]
        if len(train) < 4 or len(held) < 1:
            continue
        try:
            kde = KineticsKDE(train)
        except ValueError:
            continue
        box = UniformBox(train)
        kde_lls = [_ll(kde, s) for s in held]
        box_lls = [_ll(box, s) for s in held]
        folds[family] = {
            "held_n": len(held),
            "train_n": len(train),
            "kde_mean_ll": mean(kde_lls),
            "box_mean_ll": mean(box_lls),
            "kde_minus_box_mean": mean(k - b for k, b in zip(kde_lls, box_lls)),
        }
    return folds


def main() -> None:
    corpus = load_corpus()
    families = {s.family for s in corpus}
    print(f"corpus: {len(corpus)} parts across {len(families)} families {sorted(families)}\n")

    loo = leave_one_out(corpus)
    print("== leave-one-out (within-corpus): does the density generalize to unseen parts? ==")
    print(f"  KDE held-out mean ll {loo['kde_mean_ll']:+.3f} | box {loo['box_mean_ll']:+.3f}")
    print(
        f"  KDE − box: {loo['kde_minus_box_mean']:+.3f} [95% CI {loo['ci_low']:+.3f}, {loo['ci_high']:+.3f}] "
        f"| KDE wins {loo['kde_wins_frac']:.0%} of parts"
    )

    lofo = leave_one_family_out(corpus)
    print("\n== leave-one-family-out (cross-family): does it generalize to an unseen family? ==")
    for family, f in lofo.items():
        print(
            f"  hold out {family:13s} (n={f['held_n']:2d}): KDE {f['kde_mean_ll']:+.3f} | "
            f"box {f['box_mean_ll']:+.3f} | KDE − box {f['kde_minus_box_mean']:+.3f}"
        )

    loo_pass = loo["ci_low"] > 0
    lofo_pass = bool(lofo) and all(f["kde_minus_box_mean"] > 0 for f in lofo.values())
    if loo_pass and lofo_pass:
        verdict = "learned manifold beats hand-set ranges within and across families: worth claiming"
    elif loo_pass:
        verdict = "density is real (wins within-corpus) but family-specific (loses cross-family): grow the corpus"
    else:
        verdict = "learned density does not beat hand-set ranges: catalog-grounded ranges suffice"
    print(f"\nverdict: {verdict}")

    result = {
        "corpus_size": len(corpus),
        "families": sorted(families),
        "leave_one_out": loo,
        "leave_one_family_out": lofo,
        "loo_beats_box_ci_excludes_zero": loo_pass,
        "lofo_beats_box_all_families": lofo_pass,
        "verdict": verdict,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
