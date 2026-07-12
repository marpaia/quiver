"""Realizable kinetic ranges for parameter tuning, grounded in the Cello characterization corpus.

Unbounded parameter search can match a target dose-response with kinetics no real part achieves ("fit on
paper"): a runaway Hill coefficient, or a dynamic range beyond anything measured. These bounds clamp a tuned
parameterization to the *dimensionless* envelope of characterized parts, the Hill coefficient and the
achievable dynamic range, read from the Cello UCF corpus (``research/quiver/data/cello``): Hill ``n`` spans
1.15 to 4.48 and dynamic range reaches ~3.4e3 across 63 gates and 21 sensors.

Only unit-free quantities are clamped. Absolute expression levels and the half-max ``K`` are in the parts'
own (RPU) units; mapping those to LOICA's parameter space is unfinished (see
``notes/physically-grounded-policy.md``), so they are left unbounded here rather than clamped against a
mismatched unit system.
"""

from __future__ import annotations

from typing import cast

from quiver.grn import NodeKind, ParamValue

# Realizable Hill cooperativity, from the Cello corpus (observed 1.15..4.48), with slight headroom.
HILL_N_RANGE: tuple[float, float] = (1.0, 5.0)

# Largest realizable fold change (max/min expression), from the Cello corpus (sensors reach ~3.4e3).
MAX_DYNAMIC_RANGE: float = 3.0e3


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_n(value: ParamValue) -> ParamValue:
    lo, hi = HILL_N_RANGE
    if isinstance(value, list):
        return cast(ParamValue, [_clamp_n(v) for v in value])
    if isinstance(value, (int, float)):
        return _clamp(float(value), lo, hi)
    return value


def _cap_dynamic_range(alpha: ParamValue) -> ParamValue:
    """Cap an ``alpha`` expression vector's fold change to :data:`MAX_DYNAMIC_RANGE`: no positive entry may
    exceed the smallest positive entry times the cap. Inert when the leaky floor is zero (an unbounded ratio
    by construction) or when a single level is present; recurses into the per-input vectors of a ``sum``."""
    if isinstance(alpha, list) and alpha and isinstance(alpha[0], list):
        return cast(ParamValue, [_cap_dynamic_range(sub) for sub in alpha])
    if isinstance(alpha, list):
        positive = [x for x in alpha if isinstance(x, (int, float)) and x > 0.0]
        if len(positive) >= 2:
            ceil = min(positive) * MAX_DYNAMIC_RANGE
            capped = [min(float(x), ceil) if isinstance(x, (int, float)) and x > 0.0 else x for x in alpha]
            return cast(ParamValue, capped)
    return alpha


def clamp_realizable(kind: NodeKind, params: dict[str, ParamValue]) -> dict[str, ParamValue]:
    """Clamp the dimensionless kinetic parameters of a node to the realizable envelope. Returns a new dict;
    parameters outside the clamped set (``K``, levels, degradation) pass through unchanged."""
    out = dict(params)
    if "n" in out:
        out["n"] = _clamp_n(out["n"])
    if "alpha" in out:
        out["alpha"] = _cap_dynamic_range(out["alpha"])
    return out
