"""Held-out failure models and alternative behavior detectors used to validate the shipped strategy.

These are not part of the default sensor design path: intrinsic-noise (:mod:`noise`) and shared-resource
(:mod:`resource_battery`) failure batteries and the context-shock battery (:mod:`stress`) behind the
decorrelation study; switch/oscillator detectors (:mod:`dynamics`) for other device classes; the
measured-curve objective (:mod:`reference`); and simulation-free structural certificates
(:mod:`certificate`). They import the shipped core, never the reverse.
"""

from __future__ import annotations

from quiver.research.objective.certificate import (
    certifies,
    is_monotone,
    loop_signs,
    monotone_gate,
    signed_interaction_graph,
)
from quiver.research.objective.dynamics import (
    envelope_decay,
    hysteresis,
    is_sustained,
    sustained_oscillation,
    ultrasensitive_switch,
)
from quiver.research.objective.noise import noise_battery, score_under_noise
from quiver.research.objective.reference import (
    ReferenceCurve,
    curve_match,
    reference_from_measurements,
    result_to_reference,
)
from quiver.research.objective.resource_battery import (
    ResourceCondition,
    apply_resource_competition,
    resource_battery,
    score_under_resource,
)
from quiver.research.objective.stress import Context, apply_context, context_battery, score_under_context

__all__ = [
    "Context",
    "ReferenceCurve",
    "ResourceCondition",
    "apply_context",
    "apply_resource_competition",
    "certifies",
    "context_battery",
    "curve_match",
    "envelope_decay",
    "hysteresis",
    "is_monotone",
    "is_sustained",
    "loop_signs",
    "monotone_gate",
    "noise_battery",
    "reference_from_measurements",
    "resource_battery",
    "result_to_reference",
    "score_under_context",
    "score_under_noise",
    "score_under_resource",
    "signed_interaction_graph",
    "sustained_oscillation",
    "ultrasensitive_switch",
]
