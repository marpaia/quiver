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

from quiver.gfn.env import initial_sensor
from quiver.gfn.structured_env import (
    MAX_REG,
    STRUCTURED_ACTIONS,
    apply_structured_action,
    available_structured_actions,
)
from quiver.grn import GRN, Node, ParamValue, is_operator
from quiver.learn._core import vargrad_step
from quiver.objective import Behavior, ObjectiveFn, score_once

nn = torch.nn

# Node-kind vocabulary, fixed so the one-hot is stable across circuits.
_KINDS: tuple[str, ...] = ("regulator", "reporter", "supplement", "source", "receiver", "hill1", "hill2", "sum")
# One-hot(kind) + [is_operator, sign, log10(K), n/5, log10(dynamic range), degradation, construction progress].
_N_FEATURES = len(_KINDS) + 7

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


def graph_tensors(grn: GRN, step: int, max_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Encode a partial circuit as ``(node features, forward adjacency, backward adjacency, regulator rows)``.

    Forward adjacency mean-aggregates a node's successors, backward its predecessors, so directed message
    passing sees regulatory direction. ``regulator rows`` lists the regulator node indices in construction
    order, the indexing the per-regulator action head addresses."""
    nodes = list(grn.nodes)
    row = {node.id: i for i, node in enumerate(nodes)}
    features = torch.tensor([_node_features(node, step, max_steps) for node in nodes], dtype=torch.float32)
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
    """A message-passing GNN over the circuit graph producing a logit per structured action."""

    def __init__(self, in_features: int = _N_FEATURES, hidden: int = 32, layers: int = 3, n_reg: int = MAX_REG) -> None:
        super().__init__()
        self.embed = nn.Linear(in_features, hidden)
        self.layers = nn.ModuleList([_MessagePassing(hidden) for _ in range(layers)])
        self.stop_head = nn.Linear(hidden, 1)
        self.relay_head = nn.Linear(hidden, 1)
        self.rewire_head = nn.Linear(hidden, 1)
        self.n_reg = n_reg

    def forward(
        self, features: torch.Tensor, forward: torch.Tensor, backward: torch.Tensor, regulator_rows: list[int]
    ) -> torch.Tensor:
        h = torch.relu(self.embed(features))
        for layer in self.layers:
            h = layer(h, forward, backward)
        stop = self.stop_head(h.mean(dim=0)).squeeze(-1)
        relay = []
        rewire = []
        for i in range(self.n_reg):
            if i < len(regulator_rows):
                node = h[regulator_rows[i]]
                relay.append(self.relay_head(node).squeeze(-1))
                rewire.append(self.rewire_head(node).squeeze(-1))
            else:
                relay.append(torch.zeros(()))
                rewire.append(torch.zeros(()))
        return torch.stack([stop, *relay, *rewire])


def _rollout(policy: GraphPolicyNet, rng: random.Random, max_steps: int) -> tuple[GRN, torch.Tensor]:
    """Sample one construction trajectory to a terminal circuit, returning it and its log-probability."""
    grn = initial_sensor()
    log_prob = torch.zeros(())
    for step in range(max_steps):
        features, forward, backward, regulator_rows = graph_tensors(grn, step, max_steps)
        logits = policy(features, forward, backward, regulator_rows)
        mask = torch.full((len(STRUCTURED_ACTIONS),), float("-inf"))
        for name in available_structured_actions(grn):
            mask[_ACTION_INDEX[name]] = 0.0
        log_probs = torch.log_softmax(logits + mask, dim=-1)
        index = int(torch.multinomial(log_probs.exp(), 1).item())
        log_prob = log_prob + log_probs[index]
        if STRUCTURED_ACTIONS[index] == "stop":
            break
        grn = apply_structured_action(grn, STRUCTURED_ACTIONS[index], rng)
    return grn, log_prob


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
) -> GraphPolicyNet:
    """Train the graph policy to sample topologies in proportion to ``reward ** beta`` (VarGrad).

    Same objective and structured action space as :func:`quiver.learn.structured.train`; only the policy
    representation changes (graph message passing over physical node features rather than an MLP over a
    five-number summary), so a run is a head-to-head architecture comparison. ``objective`` sets the reward
    (default the ``score_once`` shape-match); pass :func:`quiver.objective.buildable_objective` to train
    against the buildability-aware signal."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = GraphPolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = []
        log_rewards = []
        for _ in range(batch):
            grn, log_prob = _rollout(policy, rng, max_steps)
            reward = objective(grn) if objective is not None else score_once(grn, behavior)
            log_probs.append(log_prob)
            log_rewards.append(beta * torch.log(torch.tensor(max(reward, 0.0) + floor)))
        vargrad_step(optimizer, log_probs, log_rewards)
    return policy


def sample_terminals(policy: GraphPolicyNet, n: int, *, max_steps: int = 8, seed: int = 0) -> list[GRN]:
    """Sample ``n`` terminal circuits from a trained graph policy (no gradient)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out: list[GRN] = []
    with torch.no_grad():
        for _ in range(n):
            grn, _ = _rollout(policy, rng, max_steps)
            out.append(grn)
    return out
