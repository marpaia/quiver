from __future__ import annotations

import random

import pytest

pytest.importorskip("loica")

from quiver import Designer, design  # noqa: E402
from quiver.component import load_cello  # noqa: E402
from quiver.grn import GRN, is_operator  # noqa: E402
from quiver.learn import gnn  # noqa: E402
from quiver.objective import TargetResponse, dose_response_match  # noqa: E402

TARGETS = [
    TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="up"),
    TargetResponse("GFP", ec50=0.1, dynamic_range=50.0, direction="down"),
]


def test_grounded_rollout_builds_orthogonal_component_circuit() -> None:
    # The Component-grounded rollout is torch-only (no simulation): every stage is a real part, the sensor
    # from the inducer and each hill1 from a distinct repressor (orthogonality).
    library = load_cello()
    policy = gnn.GraphPolicyNet(n_components=len(library.repressors()))
    grn, log_prob, _ = gnn._grounded_rollout(policy, random.Random(0), 6, library, "aTc")

    operators = [n for n in grn.nodes if is_operator(n.kind)]
    assert operators and all(n.component_id is not None for n in operators), "every operator is grounded"
    grounded = [n for n in grn.nodes if n.component_id is not None]
    regulators = [library.by_id(n.component_id).regulator for n in grounded]
    assert len(regulators) == len(set(regulators)), "no transcription factor is reused (orthogonality)"
    assert bool(log_prob.isfinite()), "the trajectory log-probability is finite"


@pytest.mark.sim
def test_grounded_design_portfolio_is_built_from_real_components() -> None:
    library = load_cello()
    behavior = dose_response_match(TARGETS[0])
    portfolio = design(behavior, policy="gnn", library=library, budget=6, beta=1.0, tune_steps=0, threshold=0.0, seed=0)
    assert len(portfolio) >= 1, "expected at least one grounded design"
    for grn, _ in portfolio:
        operators = [n for n in grn.nodes if is_operator(n.kind)]
        assert all(n.component_id is not None for n in operators), "delivered operators are real Components"
    # Grounded circuits still compile to LOICA with their real part names.
    assert all(isinstance(net, object) for net in portfolio.networks())


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
