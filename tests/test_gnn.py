"""The graph policy: it reads a part-built circuit as a graph, behind the same conditioned interface."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("loica")

from quiver.component import load_cello  # noqa: E402
from quiver.grn import is_operator  # noqa: E402
from quiver.learn import gnn  # noqa: E402
from quiver.learn.amortized import Brief  # noqa: E402
from quiver.objective import inducible_response  # noqa: E402


def _briefs() -> list[Brief]:
    return [
        Brief(encoding=(1.0, 0.0, 0.5), inducer="aTc", behavior=inducible_response("GFP", direction="up")),
        Brief(encoding=(0.0, 1.0, 0.25), inducer="aTc", behavior=inducible_response("GFP", direction="down")),
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


def test_conditioning_width_sizes_the_policy() -> None:
    node = gnn.conditioned_policy(4, cond_mode="node")
    head = gnn.conditioned_policy(4, cond_mode="head")
    # "node" widens the per-node input; "head" leaves message passing on physical features alone.
    assert node.embed.in_features == head.embed.in_features + 4
    assert head.cond_dim == 4 and node.cond_dim == 0


@pytest.mark.sim
def test_conditioned_graph_policy_samples_every_brief() -> None:
    briefs = _briefs()
    policy = gnn.train_conditioned(briefs, steps=2, batch=2, beta=1.0, seed=0)
    for brief in briefs:
        circuits = gnn.sample_conditioned(policy, brief, 3, seed=1)
        assert len(circuits) == 3
        assert all(c.topology_hash() for c in circuits)


@pytest.mark.sim
def test_head_conditioned_graph_policy_samples() -> None:
    briefs = _briefs()
    policy = gnn.train_conditioned(briefs, steps=2, batch=2, beta=1.0, seed=0, cond_mode="head")
    assert policy.cond_dim == len(briefs[0].encoding)
    circuits = gnn.sample_conditioned(policy, briefs[0], 3, seed=1, cond_mode="head")
    assert len(circuits) == 3
