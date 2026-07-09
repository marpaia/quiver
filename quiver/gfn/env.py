"""The constructive MDP a GFlowNet samples over: build a sensor topology one move at a time.

A state is a partial sensor circuit; actions grow or rewire it; ``stop`` emits it as a terminal design.
The moves are the same structural vocabulary the evolutionary search uses (`quiver.generate.topology`),
so the learned sampler and the baseline explore the same topology space, the fair-comparison condition.
This module is torch-free; the policy and trainer that consume it live in ``quiver.learn.gflownet``.
"""

from __future__ import annotations

import random

from quiver.generate.topology import SENSOR_MOVE_NAMES, apply_sensor_move
from quiver.grn import GRN, Edge, Node

# The action set: the structural moves, plus terminating the construction.
ACTIONS: tuple[str, ...] = SENSOR_MOVE_NAMES + ("stop",)


def initial_sensor() -> GRN:
    """The minimal buildable sensor: an inducer drives a receiver that expresses a regulator, which a
    Hill1 promoter reads to drive the reporter. One regulator, so a ``relay`` move has something to graft
    from; a second regulator (from a relay) unlocks ``rewire``.
    """
    return GRN(
        nodes=(
            Node("sup", "supplement", "aTc", {}),
            Node("op0", "receiver", "op0", {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}),
            Node("reg0", "regulator", "reg0", {"degradation_rate": 1.0, "init_concentration": 0.0}),
            Node("op1", "hill1", "op1", {"alpha": [0.0, 100.0], "K": 1.0, "n": 2.0}),
            Node("out", "reporter", "GFP", {"signal_id": "gfp", "degradation_rate": 0.0, "init_concentration": 0.0}),
        ),
        edges=(Edge("sup", "op0"), Edge("op0", "reg0"), Edge("reg0", "op1"), Edge("op1", "out")),
    )


def available_actions(grn: GRN) -> list[str]:
    """The moves valid from ``grn``: ``relay`` and ``stop`` always; ``rewire`` once there are at least
    two regulators to point an output input at."""
    regulators = sum(1 for node in grn.nodes if node.kind == "regulator")
    actions = ["relay", "stop"]
    if regulators >= 2:
        actions.append("rewire")
    return actions


def apply_action(grn: GRN, action: str, rng: random.Random) -> GRN:
    """Apply a non-terminal action. ``stop`` is handled by the caller and returns ``grn`` unchanged."""
    if action == "stop":
        return grn
    return apply_sensor_move(grn, action, rng)
