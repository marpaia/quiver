"""A target-conditioned GFlowNet, the amortized sampler.

The single-target GFlowNet (`gflownet.py`) learns one target's design distribution. This conditions the
policy on the *target* (its EC50, dynamic range, direction), so one policy trained across a distribution
of targets can be sampled for any target, including ones held out of training. That is the amortization
claim: train once, design for many targets. Each training batch is a single target (so the VarGrad
baseline, which estimates that target's normalizer, stays correct); training cycles through the targets.
"""

from __future__ import annotations

import math
import random

import torch

from quiver.component import ComponentLibrary
from quiver.gfn.env import ACTIONS, apply_action, available_actions
from quiver.grn import GRN
from quiver.learn._core import rollout, vargrad_step
from quiver.learn.gflownet import PolicyNet, features, grounding_hooks
from quiver.objective import TargetResponse, dose_response_match, score_once

_ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
_TARGET_FEATURES = 3
_CONDITIONED_FEATURES = 5 + _TARGET_FEATURES


def target_features(target: TargetResponse) -> list[float]:
    """A fixed-size encoding of a design target for the policy: log EC50, log dynamic range, direction."""
    direction = {"up": 1.0, "down": 0.0}.get(target.direction, 0.5)
    return [
        math.log10(max(target.ec50, 1e-6)) / 3.0,
        math.log10(max(target.dynamic_range, 1.0)) / 3.0,
        direction,
    ]


def _sample_trajectory(
    policy: PolicyNet,
    target: TargetResponse,
    rng: random.Random,
    max_steps: int,
    *,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> tuple[GRN, torch.Tensor]:
    encoding = target_features(target)

    def encode(grn: GRN, step: int) -> list[float]:
        return features(grn, step, max_steps) + encoding

    initial, ground = grounding_hooks(policy, library, inducer, encode)
    return rollout(
        policy,
        actions=ACTIONS,
        action_index=_ACTION_INDEX,
        available=available_actions,
        apply=apply_action,
        encode=encode,
        rng=rng,
        max_steps=max_steps,
        initial=initial,
        ground=ground,
    )


def train_conditioned(
    targets: list[TargetResponse],
    *,
    steps: int = 60,
    batch: int = 12,
    lr: float = 1e-2,
    beta: float = 1.0,
    floor: float = 1e-3,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> PolicyNet:
    """Train one target-conditioned policy across ``targets`` (one target per batch, cycling). Uses
    ``steps * batch`` simulations.

    When ``library`` is given the amortized sampler is Component-grounded: one policy trained across the
    target distribution assembles every design from real parts (the sensor from ``inducer``, each stage a
    policy-chosen repressor), so a trained :class:`~quiver.portfolio.Designer` designs any target as a real
    bill-of-Components with no retraining."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_components = len(library.gates()) if library is not None else 0
    policy = PolicyNet(in_features=_CONDITIONED_FEATURES, n_components=n_components)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(steps):
        target = targets[step % len(targets)]
        behavior = dose_response_match(target)
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, target, rng, max_steps, library=library, inducer=inducer)
            reward = score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_conditioned(
    policy: PolicyNet,
    target: TargetResponse,
    n: int,
    *,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained policy, conditioned on ``target``. With ``library`` the
    circuits are Component-grounded (must match how the policy was trained)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, target, rng, max_steps, library=library, inducer=inducer)
            out.append(grn)
    return out
