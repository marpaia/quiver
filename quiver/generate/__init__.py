"""Generator primitives for the design search: the topology construction vocabulary, the realizable-kinetics
clamp, and the local parameter-tuning step. The non-learned search baselines live in
:mod:`quiver.research.baselines`.
"""

from __future__ import annotations

from quiver.generate.optimize import tune_best, tune_score
from quiver.generate.realizable import is_realizable
from quiver.generate.search import example_inducible, mutate, proposals
from quiver.generate.topology import (
    diverse_topologies,
    random_sensor,
    sensor_cascade,
    sensor_topologies,
    structural_mutate,
)

__all__ = [
    "diverse_topologies",
    "example_inducible",
    "is_realizable",
    "mutate",
    "proposals",
    "random_sensor",
    "sensor_cascade",
    "sensor_topologies",
    "structural_mutate",
    "tune_best",
    "tune_score",
]
