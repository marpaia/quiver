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

from quiver.component import (
    ComponentLibrary,
    gate_availability,
    ground_operator,
    ground_sensor,
    ungrounded_operator_ids,
)
from quiver.generate.optimize import tune_score
from quiver.gfn.env import ACTIONS, apply_action, available_actions, initial_sensor
from quiver.grn import GRN, is_operator
from quiver.learn._core import rollout, vargrad_step
from quiver.objective import Behavior, ObjectiveFn, score_once

nn = torch.nn

_ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
_N_FEATURES = 5

# A function mapping a partial circuit and step to the policy's feature vector.
Encoder = Callable[[GRN, int], list[float]]
# A grounding hook: ground any new stage in a Component, returning the grounded circuit and the log-probability
# of the Component choices.
Grounder = Callable[[GRN], tuple[GRN, torch.Tensor]]


def features(grn: GRN, step: int, max_steps: int) -> list[float]:
    """A fixed-size summary of a partial circuit for the policy: regulator and operator counts, whether a
    two-input logic stage is present, edge count, and construction progress."""
    regulators = sum(1 for node in grn.nodes if node.kind == "regulator")
    operators = sum(1 for node in grn.nodes if is_operator(node.kind))
    has_logic = 1.0 if any(node.kind == "hill2" for node in grn.nodes) else 0.0
    return [regulators / 8.0, operators / 8.0, has_logic, len(grn.edges) / 16.0, step / max_steps]


class PolicyNet(nn.Module):
    """A small MLP over circuit features producing a logit per construction action.

    With ``n_components > 0`` it also carries a Component head over a fixed Component vocabulary, read from the
    same feature summary, so the feature-summary policy can ground each stage in a real part; ``n_components
    == 0`` leaves it abstract."""

    def __init__(
        self,
        in_features: int = _N_FEATURES,
        hidden: int = 32,
        n_actions: int = len(ACTIONS),
        n_components: int = 0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.n_components = n_components
        if n_components > 0:
            self.component_net = nn.Sequential(
                nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_components)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits

    def component_logits(self, x: torch.Tensor) -> torch.Tensor:
        """A logit per Component in the vocabulary, from the circuit's feature summary."""
        logits: torch.Tensor = self.component_net(x)
        return logits


def make_grounder(policy: PolicyNet, library: ComponentLibrary, encode: Encoder) -> Grounder:
    """A grounding hook for :func:`quiver.learn._core.rollout`: ground every not-yet-grounded ``hill1`` stage
    by sampling a gate (repressor or activator) from ``policy``'s Component head, masked so no transcription
    factor is reused (orthogonality). Returns the grounded circuit and the summed log-probability of the
    Component choices."""
    reps = library.gates()

    def ground(grn: GRN) -> tuple[GRN, torch.Tensor]:
        log_prob = torch.zeros(())
        for node_id in ungrounded_operator_ids(grn, "hill1"):
            available = gate_availability(grn, library)
            if not any(available):
                break
            feats = torch.tensor(encode(grn, 0), dtype=torch.float32)
            logits = policy.component_logits(feats)
            mask = torch.tensor([0.0 if ok else float("-inf") for ok in available])
            log_probs = torch.log_softmax(logits + mask, dim=-1)
            index = int(torch.multinomial(log_probs.exp(), 1).item())
            log_prob = log_prob + log_probs[index]
            grn = ground_operator(grn, node_id, reps[index])
        return grn, log_prob

    return ground


def grounding_hooks(
    policy: PolicyNet, library: ComponentLibrary | None, inducer: str, encode: Encoder
) -> tuple[Callable[[], GRN] | None, Grounder | None]:
    """The ``(initial, ground)`` hooks passed to :func:`quiver.learn._core.rollout`: with a ``library`` the
    initial circuit has its sensor grounded and each stage is grounded via ``policy``'s Component head;
    ``(None, None)`` for an abstract rollout."""
    if library is None:
        return None, None

    def initial() -> GRN:
        return ground_sensor(initial_sensor(), library, inducer)

    return initial, make_grounder(policy, library, encode)


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
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> PolicyNet:
    """Train a policy to sample topologies in proportion to ``reward ** beta`` (VarGrad). Uses
    ``steps * batch`` reward evaluations.

    ``objective`` sets the reward: given, it is a circuit-to-scalar (e.g.
    :func:`quiver.objective.buildable_objective`, robustness discounted by burden) evaluated per candidate;
    absent, the bare ``score_once`` shape-match. Note a robustness objective costs ``ensemble`` simulations
    per evaluation, so training simulation count scales accordingly.

    ``gate`` optionally multiplies each candidate's reward by a topology-dependent factor before tempering,
    steering the policy toward topologies a structural certificate favors.

    When ``library`` is given the sampler is *Component-grounded*: every stage is built from a real SBOL
    Component (the sensor from ``inducer``, each ``hill1`` from a policy-chosen gate --- repressor or
    activator --- orthogonally), and the reward is scored on the grounded circuit."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_components = len(library.gates()) if library is not None else 0
    policy = PolicyNet(n_components=n_components)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _sample_trajectory(policy, rng, max_steps, library=library, inducer=inducer)
            reward = objective(grn) if objective is not None else score_once(grn, behavior)
            if gate is not None:
                reward *= gate(grn)
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
    """Sample ``n`` terminal circuits from a trained policy (no gradient). With ``library`` the circuits are
    Component-grounded (must match how the policy was trained)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _sample_trajectory(policy, rng, max_steps, library=library, inducer=inducer)
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
