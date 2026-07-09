from __future__ import annotations

import random

from quiver.gfn import ACTIONS, apply_action, available_actions, initial_sensor
from quiver.grn import is_operator

_EXPECTED_INPUTS = {"source": 0, "receiver": 1, "hill1": 1, "hill2": 2}


def test_initial_sensor_is_a_valid_one_regulator_circuit() -> None:
    grn = initial_sensor()
    kinds = {n.kind for n in grn.nodes}
    assert {"supplement", "receiver", "regulator", "reporter"} <= kinds
    assert available_actions(grn) == ["relay", "stop"]  # rewire needs a second regulator


def test_actions_keep_the_circuit_a_valid_sensor() -> None:
    rng = random.Random(0)
    grn = initial_sensor()
    for _ in range(40):
        action = rng.choice([a for a in ACTIONS if a in available_actions(grn)])
        grn = apply_action(grn, action, rng)
        assert any(e.target == n.id for n in grn.nodes if n.kind == "reporter" for e in grn.edges)
        for op in (n for n in grn.nodes if is_operator(n.kind)):
            assert len(grn.inputs_of(op.id)) == _EXPECTED_INPUTS[op.kind]
