"""A graph neural network policy over the bipartite circuit graph.

The MLP policy of :mod:`quiver.learn.gflownet` reads a five-number summary of a partial circuit, so the
kinetics, the signed wiring, and per-part identity are invisible to it. This policy instead reads the
circuit *as a graph*: every species and operator node carries its physical features (kind, activation sign,
Hill ``K``/``n``, dynamic range, degradation), and directed message passing over the bipartite wiring builds
node embeddings. A graph neural network is a relational inductive bias, the physics enters through the
representation (see ``notes/physically-grounded-policy.md``).

The per-node embeddings map onto the *structured* action space (:mod:`quiver.gfn.structured_env`): a
per-regulator ``relay``/``rewire`` logit read from that regulator's embedding, plus a graph-level ``stop``
logit, so the policy chooses which regulator to act on directly. Training is the same VarGrad objective as
the MLP policy; only the representation and action head change, so the two are a head-to-head architecture
comparison under an identical reward.
"""

from __future__ import annotations

import math
import random

import torch

from quiver.component import (
    ComponentLibrary,
    ground_operator,
    ground_sensor,
    repressor_availability,
    ungrounded_operator_ids,
)
from quiver.gfn.env import initial_sensor
from quiver.gfn.structured_env import MAX_REG, STRUCTURED_ACTIONS, apply_structured_action, available_structured_actions
from quiver.grn import GRN, Node, ParamValue, is_operator
from quiver.learn._core import vargrad_step
from quiver.learn.amortized import target_features
from quiver.objective import Behavior, ObjectiveFn, TargetResponse, dose_response_match, score_once

nn = torch.nn

# Node-kind vocabulary, fixed so the one-hot is stable across circuits.
_KINDS: tuple[str, ...] = ("regulator", "reporter", "supplement", "source", "receiver", "hill1", "hill2", "sum")
# One-hot(kind) + [is_operator, sign, log10(K), n/5, log10(dynamic range), degradation, construction progress].
_N_FEATURES = len(_KINDS) + 7
# The target encoding appended to every node when the policy is target-conditioned (log EC50, log range, dir).
_TARGET_FEATURES = 3

_ACTION_INDEX = {name: index for index, name in enumerate(STRUCTURED_ACTIONS)}


def _flat_positive(value: ParamValue) -> list[float]:
    if isinstance(value, list):
        return [x for entry in value for x in _flat_positive(entry)]
    if isinstance(value, (int, float)) and value > 0.0:
        return [float(value)]
    return []


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sign(node: Node) -> float:
    """Activation (+1) or repression (-1) of a single-input operator, read from its ``alpha`` bounds."""
    alpha = node.params.get("alpha")
    if node.kind in ("receiver", "hill1") and isinstance(alpha, list) and len(alpha) >= 2:
        lo, hi = alpha[0], alpha[-1]
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return 1.0 if hi > lo else (-1.0 if hi < lo else 0.0)
    return 0.0


def _node_features(node: Node, step: int, max_steps: int) -> list[float]:
    """The physical feature vector for one node: identity, sign, and kinetics the MLP policy cannot see."""
    one_hot = [1.0 if node.kind == kind else 0.0 for kind in _KINDS]
    k = node.params.get("K")
    log_k = (
        math.log10(k)
        if isinstance(k, (int, float)) and k > 0.0
        else _mean([math.log10(v) for v in _flat_positive(k)]) if isinstance(k, list) else 0.0
    )
    n = node.params.get("n")
    n_norm = (
        float(n)
        if isinstance(n, (int, float))
        else _mean([float(v) for v in n if isinstance(v, (int, float))]) if isinstance(n, list) else 0.0
    ) / 5.0
    positive = _flat_positive(node.params.get("alpha"))
    log_dr = math.log10(max(positive) / min(positive)) if len(positive) >= 2 and min(positive) > 0.0 else 0.0
    degradation = node.params.get("degradation_rate", 0.0)
    return one_hot + [
        1.0 if is_operator(node.kind) else 0.0,
        _sign(node),
        log_k,
        n_norm,
        log_dr,
        float(degradation) if isinstance(degradation, (int, float)) else 0.0,
        step / max_steps,
    ]


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalize an adjacency to mean-aggregate neighbors, isolated nodes contribute a zero message."""
    degree = matrix.sum(dim=1, keepdim=True)
    return matrix / degree.clamp(min=1.0)


def graph_tensors(
    grn: GRN, step: int, max_steps: int, target: TargetResponse | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Encode a partial circuit as ``(node features, forward adjacency, backward adjacency, regulator rows)``.

    Forward adjacency mean-aggregates a node's successors, backward its predecessors, so directed message
    passing sees regulatory direction. ``regulator rows`` lists the regulator node indices in construction
    order, the indexing the per-regulator action head addresses. ``target``, if given, appends its encoding
    to every node's feature row, so the conditioned policy reads the design target."""
    nodes = list(grn.nodes)
    row = {node.id: i for i, node in enumerate(nodes)}
    encoding = target_features(target) if target is not None else []
    features = torch.tensor([_node_features(node, step, max_steps) + encoding for node in nodes], dtype=torch.float32)
    adjacency = torch.zeros((len(nodes), len(nodes)))
    for edge in grn.edges:
        if edge.source in row and edge.target in row:
            adjacency[row[edge.source], row[edge.target]] = 1.0
    forward = _row_normalize(adjacency)
    backward = _row_normalize(adjacency.t())
    regulator_rows = [row[node.id] for node in nodes if node.kind == "regulator"]
    return features, forward, backward, regulator_rows


class _MessagePassing(nn.Module):
    """One directed message-passing layer: a node updates from itself, its predecessors, and its successors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.in_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
        message = self.self_lin(h) + self.in_lin(backward @ h) + self.out_lin(forward @ h)
        return torch.relu(message)


class GraphPolicyNet(nn.Module):
    """A message-passing GNN over the circuit graph producing a logit per structured action.

    ``cond_dim > 0`` conditions the *action heads* on an external vector (a design target): the vector is
    concatenated to each regulator embedding and to the graph readout right before the heads. This is an
    alternative to appending the target to every node's input features, where message passing dilutes a
    constant, structure-independent signal; conditioning the heads keeps it undiluted at the decision."""

    def __init__(
        self,
        in_features: int = _N_FEATURES,
        hidden: int = 32,
        layers: int = 3,
        n_reg: int = MAX_REG,
        cond_dim: int = 0,
        n_components: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_features, hidden)
        self.layers = nn.ModuleList([_MessagePassing(hidden) for _ in range(layers)])
        self.stop_head = nn.Linear(hidden + cond_dim, 1)
        self.relay_head = nn.Linear(hidden + cond_dim, 1)
        self.rewire_head = nn.Linear(hidden + cond_dim, 1)
        self.n_reg = n_reg
        self.cond_dim = cond_dim
        # A graph-level head over a fixed Component vocabulary, chosen when a stage is grounded in a real
        # part; ``n_components == 0`` leaves the policy abstract (no grounding).
        self.n_components = n_components
        if n_components > 0:
            self.component_head = nn.Linear(hidden + cond_dim, n_components)

    def component_logits(
        self,
        features: torch.Tensor,
        forward: torch.Tensor,
        backward: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """A logit per Component in the vocabulary, read from the graph-level readout of the partial circuit
        (the same message passing as the structural heads), for grounding a stage in a real part."""
        h = torch.relu(self.embed(features))
        for layer in self.layers:
            h = layer(h, forward, backward)
        readout = h.mean(dim=0)
        if cond is not None:
            readout = torch.cat([readout, cond])
        logits: torch.Tensor = self.component_head(readout)
        return logits

    def forward(
        self,
        features: torch.Tensor,
        forward: torch.Tensor,
        backward: torch.Tensor,
        regulator_rows: list[int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = torch.relu(self.embed(features))
        for layer in self.layers:
            h = layer(h, forward, backward)

        def with_cond(vector: torch.Tensor) -> torch.Tensor:
            return torch.cat([vector, cond]) if cond is not None else vector

        stop = self.stop_head(with_cond(h.mean(dim=0))).squeeze(-1)
        relay = []
        rewire = []
        for i in range(self.n_reg):
            if i < len(regulator_rows):
                node = with_cond(h[regulator_rows[i]])
                relay.append(self.relay_head(node).squeeze(-1))
                rewire.append(self.rewire_head(node).squeeze(-1))
            else:
                relay.append(torch.zeros(()))
                rewire.append(torch.zeros(()))
        return torch.stack([stop, *relay, *rewire])


def _rollout(
    policy: GraphPolicyNet,
    rng: random.Random,
    max_steps: int,
    target: TargetResponse | None = None,
    cond_mode: str = "node",
) -> tuple[GRN, torch.Tensor, torch.Tensor]:
    """Sample one construction trajectory to a terminal circuit, returning it, its log-probability, and the
    summed per-step policy entropy (for the optional entropy regularizer). ``target`` conditions the policy;
    ``cond_mode`` selects how: ``"node"`` appends the target to node input features, ``"head"`` feeds it to
    the action heads (undiluted by message passing)."""
    head_cond = target is not None and cond_mode == "head"
    cond = torch.tensor(target_features(target), dtype=torch.float32) if head_cond and target else None
    feature_target = None if cond_mode == "head" else target
    grn = initial_sensor()
    log_prob = torch.zeros(())
    entropy = torch.zeros(())
    for step in range(max_steps):
        features, forward, backward, regulator_rows = graph_tensors(grn, step, max_steps, feature_target)
        logits = policy(features, forward, backward, regulator_rows, cond)
        mask = torch.full((len(STRUCTURED_ACTIONS),), float("-inf"))
        for name in available_structured_actions(grn):
            mask[_ACTION_INDEX[name]] = 0.0
        log_probs = torch.log_softmax(logits + mask, dim=-1)
        probs = log_probs.exp()
        entropy = entropy - (torch.where(probs > 0, log_probs, torch.zeros_like(log_probs)) * probs).sum()
        index = int(torch.multinomial(probs, 1).item())
        log_prob = log_prob + log_probs[index]
        if STRUCTURED_ACTIONS[index] == "stop":
            break
        grn = apply_structured_action(grn, STRUCTURED_ACTIONS[index], rng)
    return grn, log_prob, entropy


def _ground_pending(
    policy: GraphPolicyNet,
    grn: GRN,
    library: ComponentLibrary,
    feature_target: TargetResponse | None,
    cond: torch.Tensor | None,
) -> tuple[GRN, torch.Tensor]:
    """Ground every not-yet-grounded ``hill1`` stage in ``grn`` by sampling a repressor Component from the
    policy's Component head, masked so a transcription factor is never reused (orthogonality). Returns the
    grounded circuit and the summed log-probability of the Component choices, so the trajectory balance sees
    the assignment as part of the forward policy."""
    reps = library.repressors()
    log_prob = torch.zeros(())
    for node_id in ungrounded_operator_ids(grn, "hill1"):
        available = repressor_availability(grn, library)
        if not any(available):
            break
        features, forward, backward, _ = graph_tensors(grn, 0, 1, feature_target)
        logits = policy.component_logits(features, forward, backward, cond)
        mask = torch.tensor([0.0 if ok else float("-inf") for ok in available])
        log_probs = torch.log_softmax(logits + mask, dim=-1)
        index = int(torch.multinomial(log_probs.exp(), 1).item())
        log_prob = log_prob + log_probs[index]
        grn = ground_operator(grn, node_id, reps[index])
    return grn, log_prob


def _grounded_rollout(
    policy: GraphPolicyNet,
    rng: random.Random,
    max_steps: int,
    library: ComponentLibrary,
    inducer: str,
    target: TargetResponse | None = None,
    cond_mode: str = "node",
) -> tuple[GRN, torch.Tensor, torch.Tensor]:
    """Like :func:`_rollout`, but every stage is grounded in a real SBOL Component as it is built: the input
    sensor from ``inducer``, and each ``hill1`` stage (the initial one and every relayed one) by a policy
    Component choice. The trajectory log-probability sums the structural and Component choices, so VarGrad
    trains the joint topology-and-assignment sampler; the reward is scored on the grounded circuit."""
    head_cond = target is not None and cond_mode == "head"
    cond = torch.tensor(target_features(target), dtype=torch.float32) if head_cond and target else None
    feature_target = None if cond_mode == "head" else target
    grn = ground_sensor(initial_sensor(), library, inducer)
    log_prob = torch.zeros(())
    entropy = torch.zeros(())
    grn, ground_lp = _ground_pending(policy, grn, library, feature_target, cond)
    log_prob = log_prob + ground_lp
    for step in range(max_steps):
        features, forward, backward, regulator_rows = graph_tensors(grn, step, max_steps, feature_target)
        logits = policy(features, forward, backward, regulator_rows, cond)
        mask = torch.full((len(STRUCTURED_ACTIONS),), float("-inf"))
        for name in available_structured_actions(grn):
            mask[_ACTION_INDEX[name]] = 0.0
        log_probs = torch.log_softmax(logits + mask, dim=-1)
        probs = log_probs.exp()
        entropy = entropy - (torch.where(probs > 0, log_probs, torch.zeros_like(log_probs)) * probs).sum()
        index = int(torch.multinomial(probs, 1).item())
        log_prob = log_prob + log_probs[index]
        if STRUCTURED_ACTIONS[index] == "stop":
            break
        grn = apply_structured_action(grn, STRUCTURED_ACTIONS[index], rng)
        grn, ground_lp = _ground_pending(policy, grn, library, feature_target, cond)
        log_prob = log_prob + ground_lp
    return grn, log_prob, entropy


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
    entropy_coef: float = 0.0,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> GraphPolicyNet:
    """Train the graph policy to sample topologies in proportion to ``reward ** beta`` (VarGrad).

    Same objective and structured action space as :func:`quiver.learn.structured.train`; only the policy
    representation changes (graph message passing over physical node features rather than an MLP over a
    five-number summary), so a run is a head-to-head architecture comparison. ``objective`` sets the reward
    (default the ``score_once`` shape-match); pass :func:`quiver.objective.buildable_objective` to train
    against the buildability-aware signal. ``entropy_coef`` adds a trajectory-entropy bonus that counters the
    policy concentrating on a few (regulator-cap) topologies at high ``beta``.

    When ``library`` is given the sampler is *Component-grounded*: every stage is built from a real SBOL
    Component (the sensor from ``inducer``, each ``hill1`` from a policy-chosen repressor, orthogonally), and
    the reward is scored on the grounded circuit."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_components = len(library.repressors()) if library is not None else 0
    policy = GraphPolicyNet(n_components=n_components)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        entropies = []
        for _ in range(batch):
            if library is not None:
                grn, log_prob, entropy = _grounded_rollout(policy, rng, max_steps, library, inducer)
            else:
                grn, log_prob, entropy = _rollout(policy, rng, max_steps)
            reward = objective(grn) if objective is not None else score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
            entropies.append(entropy)
        vargrad_step(optimizer, log_probs, log_rewards, entropies=entropies, entropy_coef=entropy_coef)
    return policy


def sample_terminals(
    policy: GraphPolicyNet,
    n: int,
    *,
    max_steps: int = 8,
    seed: int = 0,
    library: ComponentLibrary | None = None,
    inducer: str = "aTc",
) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained graph policy (no gradient). With ``library`` the circuits
    are Component-grounded (must match how the policy was trained)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            if library is not None:
                grn, _, _ = _grounded_rollout(policy, rng, max_steps, library, inducer)
            else:
                grn, _, _ = _rollout(policy, rng, max_steps)
            out.append(grn)
    return out


def conditioned_policy(cond_mode: str = "node") -> GraphPolicyNet:
    """A graph policy sized for target conditioning. ``"node"`` appends the target encoding to node input
    features; ``"head"`` feeds it to the action heads instead (physical features only enter message passing).
    """
    if cond_mode == "head":
        return GraphPolicyNet(in_features=_N_FEATURES, cond_dim=_TARGET_FEATURES)
    return GraphPolicyNet(in_features=_N_FEATURES + _TARGET_FEATURES)


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
    entropy_coef: float = 0.0,
    cond_mode: str = "node",
) -> GraphPolicyNet:
    """Train one target-conditioned graph policy across ``targets`` (one target per batch, cycling), the
    graph-policy analogue of :func:`quiver.learn.amortized.train_conditioned`. ``cond_mode`` selects how the
    target enters (``"node"`` into node features, ``"head"`` into the action heads); a single policy serves
    the whole distribution. Uses ``steps * batch`` simulations."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = conditioned_policy(cond_mode)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(steps):
        target = targets[step % len(targets)]
        behavior = dose_response_match(target)
        log_probs = []
        log_rewards = []
        entropies = []
        for _ in range(batch):
            grn, log_prob, entropy = _rollout(policy, rng, max_steps, target, cond_mode)
            reward = score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
            entropies.append(entropy)
        vargrad_step(optimizer, log_probs, log_rewards, entropies=entropies, entropy_coef=entropy_coef)
    return policy


def sample_conditioned(
    policy: GraphPolicyNet,
    target: TargetResponse,
    n: int,
    *,
    max_steps: int = 8,
    seed: int = 0,
    cond_mode: str = "node",
) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained graph policy, conditioned on ``target`` (no gradient).
    ``cond_mode`` must match the mode the policy was trained with."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _, _ = _rollout(policy, rng, max_steps, target, cond_mode)
            out.append(grn)
    return out
