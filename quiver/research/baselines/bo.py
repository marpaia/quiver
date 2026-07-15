"""Bayesian optimization over sensor topologies: a Gaussian-process UCB search.

Included as the comparator for the expensive-oracle framing, the setting BO is built for. Each candidate
topology is embedded in a small feature vector (regulator count, operator count, logic flag, edge count); a
Gaussian process with an RBF kernel is fit to the evaluated (features, score) pairs, and each round proposes
a pool of random topologies and evaluates the one maximizing the upper-confidence bound. It searches the
same space with the same per-candidate scoring as the other baselines and returns a ``(topology_hash,
score)`` stream in evaluation order.

Because the simulation objective here is *cheap* (BO's regime is expensive oracles) and BO optimizes for a
single high score rather than diversity, it is expected to concentrate rather than spread, a fair
negative comparator for a diversity-first metric.
"""

from __future__ import annotations

import random

import numpy as np

from quiver.generate.optimize import tune_score
from quiver.generate.topology import random_sensor
from quiver.grn import GRN, is_operator
from quiver.objective import Behavior

Scored = tuple[str, float]


def _features(grn: GRN) -> list[float]:
    """A fixed-length embedding of a topology for the GP: normalized regulator count, operator count, a
    two-input-logic flag, and edge count."""
    regulators = sum(1 for node in grn.nodes if node.kind == "regulator")
    operators = sum(1 for node in grn.nodes if is_operator(node.kind))
    has_logic = 1.0 if any(node.kind == "hill2" for node in grn.nodes) else 0.0
    return [regulators / 4.0, operators / 6.0, has_logic, len(grn.edges) / 10.0]


def _rbf(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * sq / (length**2))


def _stable_cholesky(gram: np.ndarray, noise: float) -> np.ndarray:
    """Cholesky factor of ``gram + noise * I``, escalating the jitter if near-duplicate points leave the
    kernel numerically non-positive-definite. The base ``noise`` succeeds in the non-degenerate case."""
    identity = np.eye(len(gram))
    jitter = noise
    for _ in range(6):
        try:
            return np.linalg.cholesky(gram + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(gram + jitter * identity)


def _gp_ucb(x: np.ndarray, y: np.ndarray, xs: np.ndarray, *, length: float, noise: float, kappa: float) -> np.ndarray:
    """Posterior mean + ``kappa`` * std of a zero-mean RBF GP fit to ``(x, y)``, evaluated at ``xs``."""
    chol = _stable_cholesky(_rbf(x, x, length), noise)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    cross = _rbf(x, xs, length)
    mean = cross.T @ alpha
    v = np.linalg.solve(chol, cross)
    var = np.ones(len(xs)) - (v**2).sum(0)
    std = np.sqrt(np.clip(var, 1e-9, None))
    return mean + kappa * std


def gp_ucb_search(
    behavior: Behavior,
    *,
    budget: int,
    tune_steps: int = 2,
    seed: int = 0,
    max_regulators: int = 4,
    pool: int = 32,
    kappa: float = 1.5,
    length: float = 0.5,
    noise: float = 1e-3,
    warmup: int = 5,
) -> list[Scored]:
    """GP-UCB search: after ``warmup`` random evaluations, each round embeds a pool of random candidate
    topologies, fits the GP to all evaluations so far, and evaluates the pool member with the highest
    upper-confidence bound, until the budget is spent."""
    rng = random.Random(seed)
    per = tune_steps + 1
    stream: list[Scored] = []
    xs: list[list[float]] = []
    ys: list[float] = []

    def evaluate(grn: GRN) -> None:
        score = tune_score(grn, behavior, rng, tune_steps)
        stream.append((grn.topology_hash(), score))
        xs.append(_features(grn))
        ys.append(score)

    n_evals = max(1, budget // per)
    for _ in range(min(warmup, n_evals)):
        evaluate(random_sensor(rng, max_regulators=max_regulators))

    while len(stream) < n_evals:
        candidates = [random_sensor(rng, max_regulators=max_regulators) for _ in range(pool)]
        ucb = _gp_ucb(
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray([_features(c) for c in candidates], dtype=float),
            length=length,
            noise=noise,
            kappa=kappa,
        )
        evaluate(candidates[int(np.argmax(ucb))])
    return stream
