"""A structure-controlling GFlowNet: the policy selects the moves' internal choices, so a topology is a
deterministic function of the action sequence.

Same VarGrad objective and features as :mod:`quiver.learn.gflownet`, over the indexed action space of
:mod:`quiver.gfn.structured_env`. Because the environment is now deterministic, there is no
move-randomness to marginalize: the sequence-to-topology gap closes, and the policy can concentrate the
topology distribution as the reward exponent grows.
"""

from __future__ import annotations

import random

import torch

from quiver.gfn.structured_env import STRUCTURED_ACTIONS, apply_structured_action, available_structured_actions
from quiver.grn import GRN
from quiver.learn._core import rollout, vargrad_step
from quiver.learn.gflownet import PolicyNet, features
from quiver.objective import Behavior, score_once

_ACTION_INDEX = {name: index for index, name in enumerate(STRUCTURED_ACTIONS)}


def structured_policy() -> PolicyNet:
    """A policy head sized to the indexed structural action space."""
    return PolicyNet(n_actions=len(STRUCTURED_ACTIONS))


def _sample_trajectory(policy: PolicyNet, rng: random.Random, max_steps: int) -> tuple[GRN, torch.Tensor]:
    return rollout(
        policy,
        actions=STRUCTURED_ACTIONS,
        action_index=_ACTION_INDEX,
        available=available_structured_actions,
        apply=apply_structured_action,
        encode=lambda grn, step: features(grn, step, max_steps),
        rng=rng,
        max_steps=max_steps,
    )


def train(
    behavior: Behavior,
    *,
    steps: int = 40,
    batch: int = 12,
    lr: float = 1e-2,
    beta: float = 1.0,
    floor: float = 1e-3,
    max_steps: int = 8,
    seed: int = 0,
) -> PolicyNet:
    """Train a structure-controlling policy to sample topologies in proportion to ``reward ** beta``
    (VarGrad). Uses ``steps * batch`` simulations."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = structured_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, rng, max_steps)
            reward = score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_terminals(policy: PolicyNet, n: int, *, max_steps: int = 8, seed: int = 0) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained structure-controlling policy (no gradient)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, rng, max_steps)
            out.append(grn)
    return out
