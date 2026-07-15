"""Held-out failure model: intrinsic molecular noise (stochastic simulation).

A failure model orthogonal to the deterministic ones: the circuit's behavior under intrinsic molecular
noise, from LOICA's Gillespie stochastic simulation. Each replicate is an independent stochastic realization
of the same design, so the replicates form a naturally heterogeneous, independent battery --- no ordered-
condition artifact --- and a design's failure vector over replicates measures how often noise breaks its
behavior. Because a noise-fragile design can pass every deterministic check, this geometry can catch failures
the parameter-shock and resource-competition models structurally cannot.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from quiver.grn import GRN
from quiver.objective.behavior import Behavior
from quiver.objective.score import score_once
from quiver.objective.simulate import SimConfig


def score_under_noise(grn: GRN, behavior: Behavior, seed: int, *, config: SimConfig | None = None) -> float:
    """The behavior detector on one stochastic realization of ``grn``. ``seed`` fixes numpy's global RNG,
    which LOICA's Gillespie stepper draws from, so the realization is reproducible.

    The stochastic assay defaults to a coarser dose sweep (fewer sampled doses) than the deterministic
    objective: Gillespie cost scales with the molecule counts integrated per sample, so sampling fewer doses
    bounds the cost of the expensive stochastic path without perturbing the per-dose dynamics (the sweep is
    assay sampling, not a kinetic parameter)."""
    stochastic = replace(config or SimConfig(dose_points=7), stochastic=True)
    np.random.seed(seed)
    return score_once(grn, behavior, stochastic)


def noise_battery(n: int = 12) -> tuple[int, ...]:
    """Replicate seeds for the intrinsic-noise battery: ``n`` independent stochastic realizations, each an
    orthogonal held-out condition (a design fails a condition when noise breaks its behavior there)."""
    return tuple(range(n))
