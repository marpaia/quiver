"""The structure-controlling MDP: indexed action masking, deterministic application, validity."""

from __future__ import annotations

import random

from quiver.generate.topology import _is_valid_sensor
from quiver.gfn.env import initial_sensor
from quiver.gfn.structured_env import (
    MAX_REG,
    apply_structured_action,
    available_structured_actions,
    regulators_in_order,
)


def test_initial_actions_offer_relay_but_not_rewire() -> None:
    # The initial sensor has one regulator: relay:0 is available, rewire needs a second regulator.
    actions = available_structured_actions(initial_sensor())
    assert "stop" in actions
    assert "relay:0" in actions
    assert not any(a.startswith("rewire") for a in actions)
    assert "relay:1" not in actions  # only one regulator to graft from


def test_relay_topology_is_deterministic_and_valid() -> None:
    grn = initial_sensor()
    # The topology is a deterministic function of the action, independent of the rng that sets the new
    # stage's (topology-invisible) activation parameter.
    first = apply_structured_action(grn, "relay:0", random.Random(0))
    second = apply_structured_action(grn, "relay:0", random.Random(7))
    assert first.topology_hash() == second.topology_hash()
    assert first.topology_hash() != grn.topology_hash()  # the wiring changed
    assert _is_valid_sensor(first)
    assert len(regulators_in_order(first)) == len(regulators_in_order(grn)) + 1


def test_rewire_unlocks_with_two_regulators_and_changes_wiring() -> None:
    grn = apply_structured_action(initial_sensor(), "relay:0", random.Random(0))  # now two regulators
    actions = available_structured_actions(grn)
    rewires = [a for a in actions if a.startswith("rewire")]
    assert rewires  # at least one regulator differs from the current output source
    rewired = apply_structured_action(grn, rewires[0], random.Random(0))
    assert _is_valid_sensor(rewired)
    assert rewired.topology_hash() != grn.topology_hash()


def test_relay_masked_at_regulator_cap() -> None:
    grn = initial_sensor()
    rng = random.Random(0)
    for _ in range(MAX_REG * 2):  # keep grafting; relays stop being offered at the cap
        relays = [a for a in available_structured_actions(grn) if a.startswith("relay")]
        if not relays:
            break
        grn = apply_structured_action(grn, relays[0], rng)
    assert len(regulators_in_order(grn)) <= MAX_REG
    assert not any(a.startswith("relay") for a in available_structured_actions(grn))
