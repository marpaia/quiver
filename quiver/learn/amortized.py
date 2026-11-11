"""The conditioned GFlowNet: one policy trained across a range of requests, sampled for any of them.

Amortization is what a GFlowNet is for. The policy is conditioned on an encoding of what is wanted, so one
training run covers a whole range of requests and each is then sampled cheaply --- including requests held
out of training. A single request is just the one-point case of the same machinery, so there is one sampler
rather than a conditioned one and an unconditioned one.

A :class:`Brief` is one conditioning point: how the request encodes, which inducer grounds the input sensor,
and the detector that scores the result. Each training batch is a single brief, so the VarGrad baseline,
which estimates that brief's normalizer, stays correct; training cycles through them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from quiver.component import ComponentLibrary, gate_availability
from quiver.gfn.env import ACTIONS, apply_action, available_actions
from quiver.grn import GRN
from quiver.learn._core import rollout, vargrad_step
from quiver.learn.gflownet import PolicyNet, features, grounding_hooks
from quiver.objective.behavior import Behavior
from quiver.objective.score import score_once

_ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}

# The circuit-summary features every policy reads, before any conditioning is appended.
SUMMARY_FEATURES = 5


def fillable_actions(library: ComponentLibrary | None) -> Callable[[GRN], list[str]]:
    """The moves available from a state, with those the library cannot fill removed.

    A grounded design assigns a distinct transcription factor to every stage, so once the library has no
    unused gate left a further stage could only be left abstract --- unbuildable from this inventory.
    Masking the move keeps the sampler's budget on circuits that can actually be assembled.
    """

    def available(grn: GRN) -> list[str]:
        actions = available_actions(grn)
        if library is not None and not any(gate_availability(grn, library)):
            return [name for name in actions if name != "relay"]
        return actions

    return available


@dataclass(frozen=True)
class Brief:
    """One conditioning point: what is wanted, which inducer to build the input from, and how to score it."""

    encoding: tuple[float, ...]
    inducer: str
    behavior: Behavior


def conditioned_policy(conditioning: int, n_components: int = 0) -> PolicyNet:
    """A policy that reads the circuit summary plus ``conditioning`` numbers describing the request."""
    return PolicyNet(in_features=SUMMARY_FEATURES + conditioning, n_components=n_components)


def _sample_trajectory(
    policy: PolicyNet,
    brief: Brief,
    rng: random.Random,
    max_steps: int,
    *,
    library: ComponentLibrary | None = None,
) -> tuple[GRN, torch.Tensor]:
    encoding = list(brief.encoding)

    def encode(grn: GRN, step: int) -> list[float]:
        return features(grn, step, max_steps) + encoding

    initial, ground = grounding_hooks(policy, library, brief.inducer, encode)
    return rollout(
        policy,
        actions=ACTIONS,
        action_index=_ACTION_INDEX,
        available=fillable_actions(library),
        apply=apply_action,
        encode=encode,
        rng=rng,
        max_steps=max_steps,
        initial=initial,
        ground=ground,
    )


def train_conditioned(
    briefs: Sequence[Brief],
    *,
    steps: int = 60,
    batch: int = 12,
    lr: float = 1e-2,
    beta: float = 4.0,
    floor: float = 1e-3,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
) -> PolicyNet:
    """Train one conditioned policy across ``briefs`` (one brief per batch, cycling), using
    ``steps * batch`` simulations.

    With ``library`` every stage is grounded in a real Component as it is built, so one trained policy
    assembles any of its briefs into a genuine bill of parts without retraining.
    """
    if not briefs:
        raise ValueError("train_conditioned needs at least one brief")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_components = len(library.gates()) if library is not None else 0
    policy = conditioned_policy(len(briefs[0].encoding), n_components)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(steps):
        brief = briefs[step % len(briefs)]
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, brief, rng, max_steps, library=library)
            reward = score_once(grn, brief.behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_conditioned(
    policy: PolicyNet,
    brief: Brief,
    n: int,
    *,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained policy for one brief, with no retraining. ``library``
    must match the one the policy was trained with."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, brief, rng, max_steps, library=library)
            out.append(grn)
    return out
