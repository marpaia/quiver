"""Shared trajectory rollout and VarGrad step for the constructive GFlowNet samplers.

The single-target (`gflownet.py`), target-conditioned (`amortized.py`), and structure-controlling
(`structured.py`) policies differ only in their action vocabulary, feature encoding, and how the
environment applies a move; the trajectory rollout and the VarGrad loss are identical. Centralizing them
here keeps the three samplers consistent, in particular their random seeding.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import torch

from quiver.gfn.env import initial_sensor
from quiver.grn import GRN


def rollout(
    policy: torch.nn.Module,
    *,
    actions: Sequence[str],
    action_index: dict[str, int],
    available: Callable[[GRN], Sequence[str]],
    apply: Callable[[GRN, str, random.Random], GRN],
    encode: Callable[[GRN, int], list[float]],
    rng: random.Random,
    max_steps: int,
) -> tuple[GRN, torch.Tensor]:
    """Sample one construction trajectory to a terminal circuit, returning it and its log-probability under
    ``policy``. ``encode`` maps ``(grn, step)`` to the policy's feature vector, ``available`` lists the
    actions legal in a state, and ``apply`` performs a chosen non-``stop`` action."""
    grn = initial_sensor()
    log_prob = torch.zeros(())
    for step in range(max_steps):
        feats = torch.tensor(encode(grn, step), dtype=torch.float32)
        mask = torch.full((len(actions),), float("-inf"))
        for name in available(grn):
            mask[action_index[name]] = 0.0
        log_probs = torch.log_softmax(policy(feats) + mask, dim=-1)
        index = int(torch.multinomial(log_probs.exp(), 1).item())
        log_prob = log_prob + log_probs[index]
        if actions[index] == "stop":
            break
        grn = apply(grn, actions[index], rng)
    return grn, log_prob


def vargrad_step(
    optimizer: torch.optim.Optimizer,
    log_probs: list[torch.Tensor],
    log_rewards: list[torch.Tensor],
    *,
    entropies: list[torch.Tensor] | None = None,
    entropy_coef: float = 0.0,
) -> None:
    """One VarGrad update: minimize the batch variance of ``beta * log R - log P_forward``. Vanishes when
    the residual is constant across the batch, which is where the terminal marginal is reward-proportional.

    ``entropy_coef`` optionally adds a trajectory-entropy bonus (``entropies`` are the summed per-step
    policy entropies of the batch's trajectories): subtracting ``entropy_coef * mean(entropies)`` from the
    loss rewards higher-entropy sampling, a light regularizer against the policy collapsing onto a few
    topologies at high ``beta``. The default ``entropy_coef = 0`` leaves the plain VarGrad update
    bit-for-bit unchanged."""
    delta = torch.stack(log_rewards) - torch.stack(log_probs)
    loss = ((delta - delta.mean()) ** 2).mean()
    if entropies is not None and entropy_coef:
        loss = loss - entropy_coef * torch.stack(entropies).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
