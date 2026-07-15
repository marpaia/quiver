"""A constructive GFlowNet over sensor topologies, trained by the VarGrad (trajectory-balance) objective.

A small policy network reads features of a partial circuit and chooses the next construction action
(`quiver.gfn.env`); a trajectory ends at ``stop`` (or a step cap) with a terminal circuit whose reward is
its simulation score. VarGrad drives the policy toward sampling terminal circuits in proportion to
``reward ** beta``, so it spreads mass across distinct high-scoring topologies rather than concentrating
on one, the diversity-seeking behavior the baselines lack.

Construction is over action *sequences* (a tree), so the backward policy is trivial and trajectory balance
reduces to matching ``log P_forward(trajectory)`` to ``log reward`` up to a constant, estimated per batch,
which is the VarGrad form.
"""

from __future__ import annotations

import random
from typing import Callable

import torch

from quiver.generate.optimize import tune_score
from quiver.gfn.env import ACTIONS, apply_action, available_actions
from quiver.grn import GRN, is_operator
from quiver.learn._core import rollout, vargrad_step
from quiver.objective import Behavior, ObjectiveFn, score_once

nn = torch.nn

_ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
_N_FEATURES = 5


def features(grn: GRN, step: int, max_steps: int) -> list[float]:
    """A fixed-size summary of a partial circuit for the policy: regulator and operator counts, whether a
    two-input logic stage is present, edge count, and construction progress."""
    regulators = sum(1 for node in grn.nodes if node.kind == "regulator")
    operators = sum(1 for node in grn.nodes if is_operator(node.kind))
    has_logic = 1.0 if any(node.kind == "hill2" for node in grn.nodes) else 0.0
    return [regulators / 8.0, operators / 8.0, has_logic, len(grn.edges) / 16.0, step / max_steps]


class PolicyNet(nn.Module):
    """A small MLP over circuit features producing a logit per construction action."""

    def __init__(self, in_features: int = _N_FEATURES, hidden: int = 32, n_actions: int = len(ACTIONS)) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits


def _sample_trajectory(policy: PolicyNet, rng: random.Random, max_steps: int) -> tuple[GRN, torch.Tensor]:
    return rollout(
        policy,
        actions=ACTIONS,
        action_index=_ACTION_INDEX,
        available=available_actions,
        apply=apply_action,
        encode=lambda grn, step: features(grn, step, max_steps),
        rng=rng,
        max_steps=max_steps,
    )


def train(
    behavior: Behavior,
    *,
    steps: int = 60,
    batch: int = 16,
    lr: float = 1e-2,
    beta: float = 1.0,
    floor: float = 1e-3,
    max_steps: int = 8,
    seed: int = 0,
    gate: Callable[[GRN], float] | None = None,
    objective: ObjectiveFn | None = None,
) -> PolicyNet:
    """Train a policy to sample topologies in proportion to ``reward ** beta`` (VarGrad). Uses
    ``steps * batch`` reward evaluations.

    ``objective`` sets the reward: given, it is a circuit-to-scalar (e.g.
    :func:`quiver.objective.buildable_objective`, robustness discounted by burden) evaluated per candidate;
    absent, the bare ``score_once`` shape-match. Note a robustness objective costs ``ensemble`` simulations
    per evaluation, so training simulation count scales accordingly.

    ``gate`` optionally multiplies each candidate's reward by a topology-dependent factor before tempering,
    steering the policy toward topologies a structural certificate favors (see
    :func:`quiver.objective.certificate.monotone_gate`).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = PolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, rng, max_steps)
            reward = objective(grn) if objective is not None else score_once(grn, behavior)
            if gate is not None:
                reward *= gate(grn)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_terminals(policy: PolicyNet, n: int, *, max_steps: int = 8, seed: int = 0) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained policy (no gradient)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, rng, max_steps)
            out.append(grn)
    return out


def discovery_stream(
    behavior: Behavior,
    *,
    seed: int = 0,
    beta: float = 1.0,
    steps: int = 60,
    batch: int = 16,
    max_steps: int = 8,
    n_samples: int = 100,
    tune_steps: int = 2,
    sample_seed: int | None = None,
    gate: Callable[[GRN], float] | None = None,
) -> list[tuple[str, float]]:
    """Train a policy on ``behavior`` and score a discovery stream from its sampled terminals.

    Trains a GFlowNet (:func:`train`), samples ``n_samples`` terminal circuits (:func:`sample_terminals`),
    and tunes each ``tune_steps`` steps against ``behavior``, returning ``(topology_hash, score)`` in
    sampling order. This is the learned analogue of the search-stream functions in
    :mod:`quiver.generate` (a train-then-sample sampler rather than a per-query search), so a run can
    compare it head-to-head with them. ``sample_seed`` defaults to ``seed`` and seeds both the sampling and
    the per-candidate tuning; ``gate`` reshapes the training reward
    (see :func:`quiver.objective.certificate.monotone_gate`).
    """
    policy = train(behavior, steps=steps, batch=batch, beta=beta, max_steps=max_steps, seed=seed, gate=gate)
    tuning_seed = seed if sample_seed is None else sample_seed
    rng = random.Random(tuning_seed)
    terminals = sample_terminals(policy, n_samples, max_steps=max_steps, seed=tuning_seed)
    return [(grn.topology_hash(), tune_score(grn, behavior, rng, tune_steps)) for grn in terminals]
