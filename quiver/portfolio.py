"""The high-level design entry point: turn a behavior objective into a portfolio of distinct circuits.

``design(behavior)`` is the one call an end user needs. It runs a generator, keeps the best score per
distinct topology, filters to the ones that clear a pass threshold, and ranks them, returning a
:class:`Portfolio` whose circuits compile straight to LOICA. The default generator is the diversity-seeking
constructive GFlowNet; ``method="evolutionary"`` and ``"random"`` are non-learned baselines. To design many
targets without retraining, train a :class:`Designer` once and reuse it, or load a pretrained one from the
Hugging Face Hub.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from huggingface_hub import ModelHubMixin

from quiver.grn import GRN

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quiver.objective import Behavior, ObjectiveFn, TargetResponse

Design = tuple[GRN, float]


@dataclass(frozen=True)
class Portfolio:
    """A ranked set of structurally distinct, high-scoring circuits for one objective.

    Iterates as ``(circuit, score)`` pairs in descending score order; every circuit is a distinct
    topology. Compile them to LOICA with :meth:`networks`.
    """

    designs: tuple[Design, ...]

    def __iter__(self) -> Iterator[Design]:
        return iter(self.designs)

    def __len__(self) -> int:
        return len(self.designs)

    def __getitem__(self, index: int) -> Design:
        return self.designs[index]

    def __repr__(self) -> str:
        top = f", top score {self.designs[0][1]:.3f}" if self.designs else ""
        return f"Portfolio({len(self.designs)} distinct circuits{top})"

    @property
    def best(self) -> Design:
        """The highest-scoring ``(circuit, score)``."""
        return self.designs[0]

    @property
    def circuits(self) -> list[GRN]:
        """The circuits, best first."""
        return [grn for grn, _ in self.designs]

    @property
    def scores(self) -> list[float]:
        """The scores, descending."""
        return [score for _, score in self.designs]

    def networks(self) -> list[Any]:
        """Compile every circuit to a LOICA ``GeneticNetwork`` (needs the ``sim`` extra)."""
        from quiver.objective import to_loica

        return [to_loica(grn) for grn, _ in self.designs]


def design(
    behavior: "Behavior",
    *,
    budget: int = 300,
    method: str = "gflownet",
    threshold: float = 0.5,
    tune_steps: int = 2,
    seed: int = 0,
    max_regulators: int = 4,
    beta: float = 4.0,
    robust: bool = True,
    ensemble: int = 8,
    jitter: float = 0.3,
    burden_capacity: float | None = 500.0,
    objective: "ObjectiveFn | None" = None,
) -> Portfolio:
    """Design a portfolio of distinct, high-scoring circuits for ``behavior``.

    ``method`` selects the generator: ``"gflownet"`` (default, the diversity-seeking learned sampler with an
    MLP-over-features policy) trains a policy for this behavior and samples a broad portfolio from it;
    ``"gnn"`` is the same sampler with a graph-neural-network policy over the circuit graph
    (:mod:`quiver.learn.gnn`); ``"evolutionary"`` and ``"random"`` are non-learned baselines. ``budget`` is
    the generation budget; ``threshold`` is the minimum score a circuit
    must clear to enter the portfolio; ``beta`` tunes the GFlowNet reward exponent (higher sharpens toward the
    best circuits, lower spreads mass more evenly).

    By default the search optimizes a *buildability-aware* objective (:func:`quiver.objective.buildable_objective`):
    the target behavior's robustness over an ``ensemble`` of perturbed parameterizations, discounted by the
    metabolic ``burden`` its topology imposes. This is the dense signal the search follows, and it is what the
    portfolio's scores and ``threshold`` filter are read against. Set ``robust=False`` for the cheaper
    single-simulation shape-match (one simulation per candidate instead of ``ensemble``), or pass a custom
    ``objective`` (a circuit-to-scalar) to override entirely. To design many targets without retraining each
    time, train a :class:`Designer` once and reuse it.
    """
    if objective is None and robust:
        from quiver.objective import buildable_objective

        objective = buildable_objective(
            behavior, ensemble=ensemble, jitter=jitter, seed=seed, burden_capacity=burden_capacity
        )
    raw = _generate(
        behavior,
        method=method,
        budget=budget,
        tune_steps=tune_steps,
        seed=seed,
        max_regulators=max_regulators,
        beta=beta,
        objective=objective,
    )
    return _portfolio_from(raw, threshold)


def _portfolio_from(raw: list[Design], threshold: float) -> Portfolio:
    """Keep the best score per distinct topology above ``threshold``, ranked best-first."""
    best: dict[str, Design] = {}
    for grn, score in raw:
        if score < threshold:
            continue
        key = grn.topology_hash()
        if key not in best or score > best[key][1]:
            best[key] = (grn, score)
    ranked = sorted(best.values(), key=lambda design: design[1], reverse=True)
    return Portfolio(tuple(ranked))


def _generate(
    behavior: "Behavior",
    *,
    method: str,
    budget: int,
    tune_steps: int,
    seed: int,
    max_regulators: int,
    beta: float,
    objective: "ObjectiveFn | None" = None,
) -> list[Design]:
    name = method.lower()
    if name in ("evolutionary", "evo"):
        from quiver.generate import evolutionary_search_designs

        return evolutionary_search_designs(
            behavior,
            budget=budget,
            tune_steps=tune_steps,
            seed=seed,
            max_regulators=max_regulators,
            objective=objective,
        )
    if name == "random":
        from quiver.generate import random_search_designs

        return random_search_designs(
            behavior,
            budget=budget,
            tune_steps=tune_steps,
            seed=seed,
            max_regulators=max_regulators,
            objective=objective,
        )
    if name in ("gflownet", "gfn"):
        from quiver.generate import tune_best
        from quiver.learn.gflownet import sample_terminals, train

        policy = train(behavior, beta=beta, seed=seed, objective=objective)
        rng = random.Random(seed)
        return [
            tune_best(grn, behavior, rng, tune_steps, objective=objective)
            for grn in sample_terminals(policy, budget, seed=seed)
        ]
    if name == "gnn":
        from quiver.generate import tune_best
        from quiver.learn.gnn import sample_terminals as sample_gnn
        from quiver.learn.gnn import train as train_gnn

        gnn_policy = train_gnn(behavior, beta=beta, seed=seed, objective=objective)
        rng = random.Random(seed)
        return [
            tune_best(grn, behavior, rng, tune_steps, objective=objective)
            for grn in sample_gnn(gnn_policy, budget, seed=seed)
        ]
    raise ValueError(f"unknown method {method!r}; choose 'evolutionary', 'random', 'gflownet', or 'gnn'")


class Designer(ModelHubMixin):  # type: ignore[misc]
    """A trained, target-conditioned sampler: train once, design portfolios for many targets.

    This is the amortized path. Where :func:`design` runs a generator from scratch on every call,
    :meth:`Designer.train` learns one policy across a distribution of sensor targets, and each
    :meth:`design` call samples that policy for a target, cheaply and without retraining, including
    targets held out of training.

    A trained designer follows the Hugging Face convention: :meth:`save_pretrained` writes it to a
    directory, :meth:`from_pretrained` loads it from a directory or a Hub repo, and :meth:`push_to_hub`
    publishes it (all inherited from ``ModelHubMixin``).
    """

    def __init__(self, policy: Any, max_steps: int = 8) -> None:
        self.policy = policy
        self.max_steps = max_steps

    def __repr__(self) -> str:
        return f"Designer(trained, max_steps={self.max_steps})"

    @classmethod
    def train(
        cls,
        targets: "Sequence[TargetResponse]",
        *,
        steps: int = 80,
        batch: int = 12,
        beta: float = 4.0,
        max_steps: int = 8,
        seed: int = 0,
    ) -> "Designer":
        """Train a target-conditioned policy across ``targets`` (uses ``steps * batch`` simulations, once)."""
        from quiver.learn.amortized import train_conditioned

        policy = train_conditioned(list(targets), steps=steps, batch=batch, beta=beta, max_steps=max_steps, seed=seed)
        return cls(policy=policy, max_steps=max_steps)

    def design(
        self,
        target: "TargetResponse",
        *,
        budget: int = 300,
        threshold: float = 0.5,
        tune_steps: int = 2,
        seed: int = 0,
    ) -> Portfolio:
        """Design a portfolio for ``target`` by sampling the trained policy; no retraining."""
        from quiver.generate import tune_best
        from quiver.learn.amortized import sample_conditioned
        from quiver.objective import dose_response_match

        behavior = dose_response_match(target)
        rng = random.Random(seed)
        candidates = sample_conditioned(self.policy, target, budget, max_steps=self.max_steps, seed=seed)
        raw = [tune_best(grn, behavior, rng, tune_steps) for grn in candidates]
        return _portfolio_from(raw, threshold)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the policy weights and architecture into ``save_directory`` (called by save_pretrained)."""
        import json

        import torch

        torch.save(self.policy.state_dict(), save_directory / "policy.pt")
        layers = list(self.policy.net)
        config = {
            "format": "quiver-designer",
            "quiver_version": _quiver_version(),
            "max_steps": self.max_steps,
            "in_features": layers[0].in_features,
            "hidden": layers[0].out_features,
            "n_actions": layers[-1].out_features,
        }
        (save_directory / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: "str | None" = None,
        cache_dir: "str | Path | None" = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: "str | bool | None" = None,
        **kwargs: Any,
    ) -> "Designer":
        """Rebuild a designer from a resolved local directory (called by from_pretrained after download)."""
        import json

        import torch

        from quiver.learn.gflownet import PolicyNet

        directory = Path(model_id)
        config = json.loads((directory / "config.json").read_text())
        policy = PolicyNet(in_features=config["in_features"], hidden=config["hidden"], n_actions=config["n_actions"])
        policy.load_state_dict(torch.load(directory / "policy.pt", map_location="cpu"))
        policy.eval()
        return cls(policy, max_steps=config["max_steps"])


def _quiver_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("quiver")
    except PackageNotFoundError:
        return "unknown"
