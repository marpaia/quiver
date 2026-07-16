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

from quiver.component import ComponentLibrary
from quiver.gfn.structured_env import STRUCTURED_ACTIONS, apply_structured_action, available_structured_actions
from quiver.grn import GRN
from quiver.learn._core import rollout, vargrad_step
from quiver.learn.gflownet import PolicyNet, features, grounding_hooks
from quiver.objective import Behavior, ObjectiveFn, score_once

_ACTION_INDEX = {name: index for index, name in enumerate(STRUCTURED_ACTIONS)}


def structured_policy(n_components: int = 0) -> PolicyNet:
    """A policy head sized to the indexed structural action space (and Component vocabulary if grounded)."""
    return PolicyNet(n_actions=len(STRUCTURED_ACTIONS), n_components=n_components)


def _sample_trajectory(
    policy: PolicyNet,
    rng: random.Random,
    max_steps: int,
    *,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> tuple[GRN, torch.Tensor]:
    def encode(grn: GRN, step: int) -> list[float]:
        return features(grn, step, max_steps)

    initial, ground = grounding_hooks(policy, library, inducer, encode)
    return rollout(
        policy,
        actions=STRUCTURED_ACTIONS,
        action_index=_ACTION_INDEX,
        available=available_structured_actions,
        apply=apply_structured_action,
        encode=encode,
        rng=rng,
        max_steps=max_steps,
        initial=initial,
        ground=ground,
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
    objective: ObjectiveFn | None = None,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> PolicyNet:
    """Train a structure-controlling policy to sample topologies in proportion to ``reward ** beta``
    (VarGrad). Uses ``steps * batch`` reward evaluations.

    ``objective`` sets the reward: given, it is a circuit-to-scalar (e.g.
    :func:`quiver.objective.buildable_objective`, robustness discounted by burden) evaluated per candidate;
    absent, the bare ``score_once`` shape-match. This matches the signature of
    :func:`quiver.learn.gnn.train`, so the MLP and graph policies are a head-to-head architecture
    comparison under an identical reward. With ``library`` the sampler is Component-grounded, exactly as the
    graph policy is, so the comparison stays fair under grounding."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = structured_policy(len(library.gates()) if library is not None else 0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, rng, max_steps, library=library, inducer=inducer)
            reward = objective(grn) if objective is not None else score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_terminals(
    policy: PolicyNet,
    n: int,
    *,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained structure-controlling policy (no gradient). With
    ``library`` the circuits are Component-grounded (must match how the policy was trained)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, rng, max_steps, library=library, inducer=inducer)
            out.append(grn)
    return out
