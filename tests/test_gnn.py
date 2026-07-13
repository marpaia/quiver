from __future__ import annotations

import pytest

pytest.importorskip("loica")

from quiver import Designer  # noqa: E402
from quiver.grn import GRN  # noqa: E402
from quiver.learn import gnn  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402

TARGETS = [
    TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="up"),
    TargetResponse("GFP", ec50=0.1, dynamic_range=50.0, direction="down"),
]


@pytest.mark.sim
def test_unconditioned_gnn_trains_and_samples() -> None:
    behavior = dose_response_match(TARGETS[0])
    policy = gnn.train(behavior, steps=2, batch=2, beta=1.0, seed=0, entropy_coef=0.01)
    circuits = gnn.sample_terminals(policy, 3, seed=1)
    assert len(circuits) == 3
    assert all(isinstance(c, GRN) for c in circuits)


@pytest.mark.sim
def test_conditioned_gnn_samples_valid_circuits() -> None:
    policy = gnn.train_conditioned(TARGETS, steps=2, batch=2, beta=1.0, seed=0)
    circuits = gnn.sample_conditioned(policy, TARGETS[0], 3, seed=1)
    assert len(circuits) == 3
    assert all(c.topology_hash() for c in circuits)


@pytest.mark.sim
def test_head_conditioned_gnn_feeds_target_to_heads() -> None:
    # Head conditioning keeps physical features out of the node input and sizes the heads for the target.
    policy = gnn.train_conditioned(TARGETS, steps=2, batch=2, beta=1.0, seed=0, cond_mode="head")
    assert policy.cond_dim == 3
    circuits = gnn.sample_conditioned(policy, TARGETS[0], 3, seed=1, cond_mode="head")
    assert len(circuits) == 3
    assert all(c.topology_hash() for c in circuits)


@pytest.mark.sim
def test_designer_gnn_roundtrips(tmp_path) -> None:
    designer = Designer.train(TARGETS, steps=2, batch=2, beta=1.0, seed=0, arch="gnn")
    assert designer.arch == "gnn"
    designer.save_pretrained(tmp_path)
    reloaded = Designer.from_pretrained(tmp_path)
    assert reloaded.arch == "gnn"
    # Deterministic sampling: the reloaded policy designs the same portfolio.
    a = designer.design(TARGETS[0], budget=6, tune_steps=1, seed=2)
    b = reloaded.design(TARGETS[0], budget=6, tune_steps=1, seed=2)
    assert [c.topology_hash() for c in a.circuits] == [c.topology_hash() for c in b.circuits]
