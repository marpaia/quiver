"""A learned density over realizable part kinetics, fit from the Cello characterization corpus.

The A2 clamp (:mod:`quiver.generate.realizable`) bounds kinetics to a hand-set *box* (Hill ``n`` in a range,
a dynamic-range ceiling). This module learns the *shape* of the realizable region instead: a density over
``(log10 K, n, log10 dynamic_range)`` estimated from characterized parts, which captures the concentration
and the cross-parameter correlation a flat box misses. It supports the aspirational Track B contribution,
steering generation toward parameter regimes real parts actually occupy rather than merely inside a box.

Two models with a common interface (:meth:`log_likelihood`, :meth:`sample`): :class:`KineticsKDE`, a Gaussian
kernel-density estimate that follows the data (and its family structure); and :class:`UniformBox`, the
hand-set-ranges baseline (independent uniform per dimension). The go/no-go for claiming the learned manifold
is whether the KDE assigns higher *held-out* likelihood than the box (``experiments/manifold.py``).

Units: only the dimensionless axes and ``K`` in the corpus' own (RPU) units are modeled here; the RPU->LOICA
map is unfinished (see :mod:`quiver.generate.realizable`), so a circuit-space penalty over ``K`` is scoped
to parts characterized in the same units.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

from quiver.grn import GRN, is_operator

# The corpus shipped in the repo; see data/cello/README.md.
_CORPUS = Path(__file__).resolve().parents[2] / "data" / "cello" / "cello_kinetics.csv"

# A tiny density floor so an out-of-support point has a finite (very negative) log-likelihood.
_FLOOR = 1e-12


@dataclass(frozen=True)
class KineticsSample:
    """One characterized part's realizable kinetics, in the corpus' units."""

    K: float
    n: float
    dynamic_range: float
    family: str


def _transform(K: float, n: float, dynamic_range: float) -> list[float]:
    """Map raw kinetics to the modeling space: ``log10 K``, ``n``, ``log10 dynamic_range``."""
    return [math.log10(max(K, 1e-9)), n, math.log10(max(dynamic_range, 1.0))]


def load_corpus(path: Path | None = None) -> list[KineticsSample]:
    """Load the characterized parts with a full ``(K, n, dynamic_range)`` from the Cello corpus."""
    rows = list(csv.DictReader((path or _CORPUS).open()))
    out: list[KineticsSample] = []
    for row in rows:
        try:
            K, n, dr = float(row["K"]), float(row["n"]), float(row["dynamic_range"])
        except (KeyError, ValueError):
            continue
        if K > 0 and n > 0 and dr > 0 and math.isfinite(dr):
            out.append(KineticsSample(K=K, n=n, dynamic_range=dr, family=row.get("family", "")))
    return out


class DensityModel(Protocol):
    """A density over ``(K, n, dynamic_range)`` supporting log-likelihood and sampling."""

    def log_likelihood(self, K: float, n: float, dynamic_range: float) -> float: ...


class KineticsKDE:
    """A Gaussian kernel-density estimate over ``(log10 K, n, log10 dynamic_range)``.

    Follows the corpus, including its family structure (a mixture of TetR-family and CRISPRi gates and
    small-molecule sensors is multimodal, which a KDE represents and a single box does not)."""

    def __init__(self, samples: Sequence[KineticsSample]) -> None:
        if len(samples) < 3:
            raise ValueError("need at least three parts to fit a 3-D density")
        data = np.array([_transform(s.K, s.n, s.dynamic_range) for s in samples], dtype=float)
        self._kde = gaussian_kde(data.T)

    def log_likelihood(self, K: float, n: float, dynamic_range: float) -> float:
        point = np.array(_transform(K, n, dynamic_range)).reshape(3, 1)
        return float(self._kde.logpdf(point)[0])

    def sample(self, n: int, *, seed: int = 0) -> list[KineticsSample]:
        drawn = self._kde.resample(n, seed=seed).T
        return [KineticsSample(K=10.0 ** x[0], n=x[1], dynamic_range=10.0 ** x[2], family="sampled") for x in drawn]


class UniformBox:
    """The hand-set-ranges baseline: independent uniform over each dimension's observed span (in the
    modeling space), the density implied by clamping to a box. Its log-likelihood is constant inside the box
    and a floor outside, so a KDE beats it only by concentrating mass where parts actually cluster."""

    def __init__(self, samples: Sequence[KineticsSample]) -> None:
        data = np.array([_transform(s.K, s.n, s.dynamic_range) for s in samples], dtype=float)
        self._lo = data.min(axis=0)
        self._hi = data.max(axis=0)
        spans = np.clip(self._hi - self._lo, 1e-9, None)
        self._log_density = float(-np.log(spans).sum())

    def log_likelihood(self, K: float, n: float, dynamic_range: float) -> float:
        point = np.array(_transform(K, n, dynamic_range))
        inside = bool(np.all(point >= self._lo) and np.all(point <= self._hi))
        return self._log_density if inside else math.log(_FLOOR)


def _operator_kinetics(grn: GRN) -> list[tuple[float, float, float]]:
    """Each operator node's ``(K, n, dynamic_range)``, for nodes carrying all three."""
    out: list[tuple[float, float, float]] = []
    for node in grn.nodes:
        if not is_operator(node.kind):
            continue
        K, n = node.params.get("K"), node.params.get("n")
        alpha = node.params.get("alpha")
        positive = [x for x in alpha if isinstance(x, (int, float)) and x > 0.0] if isinstance(alpha, list) else []
        if isinstance(K, (int, float)) and isinstance(n, (int, float)) and len(positive) >= 2 and min(positive) > 0:
            out.append((float(K), float(n), max(positive) / min(positive)))
    return out


def circuit_log_likelihood(grn: GRN, model: DensityModel) -> float:
    """Mean manifold log-likelihood over a circuit's operator nodes (0.0 for a circuit with no scored node)."""
    kinetics = _operator_kinetics(grn)
    if not kinetics:
        return 0.0
    return sum(model.log_likelihood(K, n, dr) for K, n, dr in kinetics) / len(kinetics)
