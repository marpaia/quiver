"""The high-level design entry point: turn a behavior objective into a portfolio of distinct circuits.

``design(behavior)`` is the one call an end user needs. It trains the diversity-seeking constructive GFlowNet,
keeps the best score per distinct topology, filters to the ones that clear a pass threshold, and ranks them,
returning a :class:`Portfolio` whose circuits compile straight to LOICA. ``policy`` selects the policy
representation (``"gnn"`` or ``"mlp"``), and a ``library`` grounds every stage in a real SBOL Component. To
design many targets without retraining, train a :class:`Designer` once and reuse it, or load a pretrained one
from the Hugging Face Hub.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
from huggingface_hub import ModelHubMixin

from quiver.component import library_from_name
from quiver.generate import tune_best
from quiver.grn import GRN
from quiver.learn.amortized import sample_conditioned, train_conditioned
from quiver.learn.gflownet import PolicyNet, sample_terminals, train
from quiver.learn.gnn import GraphPolicyNet
from quiver.learn.gnn import sample_conditioned as sample_gnn_conditioned
from quiver.learn.gnn import sample_terminals as sample_gnn
from quiver.learn.gnn import train as train_gnn
from quiver.learn.gnn import train_conditioned as train_gnn_conditioned
from quiver.objective import buildable_objective, dose_response_match, to_loica

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quiver.component import ComponentLibrary
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
        """Compile every circuit to a LOICA ``GeneticNetwork``."""
        return [to_loica(grn) for grn, _ in self.designs]


def design(
    behavior: "Behavior",
    *,
    budget: int = 300,
    policy: str = "gnn",
    threshold: float = 0.5,
    tune_steps: int = 2,
    seed: int = 0,
    beta: float = 4.0,
    robust: bool = True,
    ensemble: int = 8,
    jitter: float = 0.3,
    burden_capacity: float | None = 500.0,
    objective: "ObjectiveFn | None" = None,
    library: "ComponentLibrary | None" = None,
    inducer: str = "aTc",
) -> Portfolio:
    """Design a portfolio of distinct, high-scoring circuits for ``behavior``.

    The strategy is fixed: a diversity-seeking GFlowNet trains a policy for this behavior and samples a broad,
    de-duplicated portfolio of distinct topologies from it. ``policy`` selects the *representation*, a
    methodological choice, not a different strategy: ``"gnn"`` (default) is the graph-neural-network policy
    over the circuit graph (:mod:`quiver.learn.gnn`), which discovers the broader single-target portfolio;
    ``"mlp"`` is the feature-summary policy (:mod:`quiver.learn.gflownet`). ``budget`` is the generation
    budget; ``threshold`` is the minimum score a circuit must clear to enter the portfolio; ``beta`` tunes the
    GFlowNet reward exponent (higher sharpens toward the best circuits, lower spreads mass more evenly).

    By default the search optimizes a *buildability-aware* objective (:func:`quiver.objective.buildable_objective`):
    the target behavior's robustness over an ``ensemble`` of perturbed parameterizations, discounted by the
    metabolic ``burden`` its topology imposes. This is the dense signal the search follows, and it is what the
    portfolio's scores and ``threshold`` filter are read against. Set ``robust=False`` for the cheaper
    single-simulation shape-match (one simulation per candidate instead of ``ensemble``), or pass a custom
    ``objective`` (a circuit-to-scalar) to override entirely. To design many targets without retraining each
    time, train a :class:`Designer` once and reuse it.
    """
    if objective is None and robust:
        objective = buildable_objective(
            behavior, ensemble=ensemble, jitter=jitter, seed=seed, burden_capacity=burden_capacity
        )
    raw = _generate(
        behavior,
        policy=policy,
        budget=budget,
        tune_steps=tune_steps,
        seed=seed,
        beta=beta,
        objective=objective,
        library=library,
        inducer=inducer,
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
    policy: str,
    budget: int,
    tune_steps: int,
    seed: int,
    beta: float,
    objective: "ObjectiveFn | None" = None,
    library: "ComponentLibrary | None" = None,
    inducer: str = "aTc",
) -> list[Design]:
    name = policy.lower()
    # A Component-grounded design's kinetics are the real part values, so keep them exact (no parameter
    # tuning) and score the grounded circuit directly; an abstract design still tunes its parameters.
    steps = 0 if library is not None else tune_steps
    rng = random.Random(seed)
    if name in ("gnn", "graph"):
        gnn_policy = train_gnn(behavior, beta=beta, seed=seed, objective=objective, library=library, inducer=inducer)
        candidates = sample_gnn(gnn_policy, budget, seed=seed, library=library, inducer=inducer)
        return [tune_best(grn, behavior, rng, steps, objective=objective) for grn in candidates]
    if name in ("mlp", "gflownet", "gfn"):
        mlp_policy = train(behavior, beta=beta, seed=seed, objective=objective, library=library, inducer=inducer)
        candidates = sample_terminals(mlp_policy, budget, seed=seed, library=library, inducer=inducer)
        return [tune_best(grn, behavior, rng, steps, objective=objective) for grn in candidates]
    raise ValueError(f"unknown policy {policy!r}; choose 'gnn' or 'mlp'")


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

    def __init__(
        self,
        policy: Any,
        max_steps: int = 8,
        arch: str = "mlp",
        library: "ComponentLibrary | None" = None,
        inducer: str = "aTc",
    ) -> None:
        self.policy = policy
        self.max_steps = max_steps
        self.arch = arch
        self.library = library
        self.inducer = inducer

    def __repr__(self) -> str:
        grounded = f", grounded in {self.library.name!r}" if self.library is not None else ""
        return f"Designer(trained, arch={self.arch!r}, max_steps={self.max_steps}{grounded})"

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
        arch: str = "mlp",
        library: "ComponentLibrary | None" = None,
        inducer: str = "aTc",
    ) -> "Designer":
        """Train a target-conditioned policy across ``targets`` (uses ``steps * batch`` simulations, once).

        ``arch`` selects the policy representation: ``"mlp"`` (default) is the feature-summary policy
        (:mod:`quiver.learn.amortized`); ``"gnn"`` is the graph policy over the circuit graph
        (:mod:`quiver.learn.gnn`), which reads each node's kinetics and signed wiring.

        With ``library`` the amortized policy is Component-grounded: it designs every target as a real
        bill-of-Components with no retraining. Grounding is supported for the ``"mlp"`` arch (the stronger
        amortizer); pass no ``library`` for the graph arch."""
        if library is not None and arch == "gnn":
            raise ValueError("Component-grounded amortization is currently supported only with arch='mlp'")
        if arch == "gnn":
            policy = train_gnn_conditioned(
                list(targets), steps=steps, batch=batch, beta=beta, max_steps=max_steps, seed=seed
            )
        else:
            policy = train_conditioned(
                list(targets),
                steps=steps,
                batch=batch,
                beta=beta,
                max_steps=max_steps,
                seed=seed,
                library=library,
                inducer=inducer,
            )
        return cls(policy=policy, max_steps=max_steps, arch=arch, library=library, inducer=inducer)

    def design(
        self,
        target: "TargetResponse",
        *,
        budget: int = 300,
        threshold: float = 0.5,
        tune_steps: int = 2,
        seed: int = 0,
    ) -> Portfolio:
        """Design a portfolio for ``target`` by sampling the trained policy; no retraining. A
        Component-grounded designer returns designs built from real parts, with kinetics kept exact."""
        behavior = dose_response_match(target)
        rng = random.Random(seed)
        steps = 0 if self.library is not None else tune_steps
        if self.arch == "gnn":
            candidates = sample_gnn_conditioned(self.policy, target, budget, max_steps=self.max_steps, seed=seed)
        else:
            candidates = sample_conditioned(
                self.policy,
                target,
                budget,
                max_steps=self.max_steps,
                seed=seed,
                library=self.library,
                inducer=self.inducer,
            )
        raw = [tune_best(grn, behavior, rng, steps) for grn in candidates]
        return _portfolio_from(raw, threshold)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the policy weights and architecture into ``save_directory`` (called by save_pretrained)."""
        torch.save(self.policy.state_dict(), save_directory / "policy.pt")
        config: dict[str, Any] = {
            "format": "quiver-designer",
            "quiver_version": _quiver_version(),
            "max_steps": self.max_steps,
            "arch": self.arch,
            "library": self.library.name if self.library is not None else None,
            "inducer": self.inducer,
        }
        if self.arch == "gnn":
            config.update(
                {
                    "in_features": self.policy.embed.in_features,
                    "hidden": self.policy.embed.out_features,
                    "layers": len(self.policy.layers),
                    "n_reg": self.policy.n_reg,
                    "cond_dim": self.policy.cond_dim,
                }
            )
        else:
            layers = list(self.policy.net)
            config.update(
                {
                    "in_features": layers[0].in_features,
                    "hidden": layers[0].out_features,
                    "n_actions": layers[-1].out_features,
                    "n_components": self.policy.n_components,
                }
            )
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
        directory = Path(model_id)
        config = json.loads((directory / "config.json").read_text())
        arch = config.get("arch", "mlp")
        policy: Any
        if arch == "gnn":
            policy = GraphPolicyNet(
                in_features=config["in_features"],
                hidden=config["hidden"],
                layers=config["layers"],
                n_reg=config["n_reg"],
                cond_dim=config.get("cond_dim", 0),
            )
        else:
            policy = PolicyNet(
                in_features=config["in_features"],
                hidden=config["hidden"],
                n_actions=config["n_actions"],
                n_components=config.get("n_components", 0),
            )
        policy.load_state_dict(torch.load(directory / "policy.pt", map_location="cpu"))
        policy.eval()
        library = library_from_name(config.get("library") or "")
        return cls(
            policy, max_steps=config["max_steps"], arch=arch, library=library, inducer=config.get("inducer", "aTc")
        )


def _quiver_version() -> str:
    try:
        return version("quiver")
    except PackageNotFoundError:
        return "unknown"
