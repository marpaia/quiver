"""Render the discovery-curve figure: cumulative distinct high-scoring topologies vs. simulation budget
for random, evolutionary, and GFlowNet search, as mean with a ±std band over seeds.

Shares the single-target configuration of ``qd_baselines.py`` so the figure's endpoints match the reported
distinct-design counts. Writes a PDF (and a JSON of the plotted means/stds) to the output path.

Run: ``uv run python experiments/discovery_figure.py [out.pdf]``.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import EVAL_BUDGET, PASS, TUNE_STEPS
from _gfn import gflownet_stream

from quiver.objective import Behavior, TargetResponse, dose_response_match
from quiver.research.baselines import evolutionary_search, random_search
from quiver.research.eval.metrics import discovery_curve

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
SEEDS = 10
PER = TUNE_STEPS + 1
METHODS = ("random", "evolutionary", "gflownet")
LABELS = {"random": "random search", "evolutionary": "evolutionary search", "gflownet": "GFlowNet"}
COLORS = {"random": "#9aa0a6", "evolutionary": "#4c78a8", "gflownet": "#e45756"}


def stream_for(method: str, behavior: Behavior, seed: int) -> list[tuple[str, float]]:
    if method == "random":
        return random_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed)
    if method == "evolutionary":
        return evolutionary_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed)
    return gflownet_stream(behavior, seed)


def curve(stream: list[tuple[str, float]]) -> tuple[int, ...]:
    hashes = [h for h, _ in stream]
    passed = [s >= PASS for _, s in stream]
    return discovery_curve(hashes, passed=passed)


def aggregate(curves: list[tuple[int, ...]]) -> tuple[list[int], list[float], list[float]]:
    """Mean, std, and simulation-budget x-axis over seed curves, truncated to their common length."""
    length = min(len(c) for c in curves)
    xs = [(i + 1) * PER for i in range(length)]
    means: list[float] = []
    stds: list[float] = []
    for i in range(length):
        column = [c[i] for c in curves]
        m = sum(column) / len(column)
        var = sum((v - m) ** 2 for v in column) / len(column)
        means.append(m)
        stds.append(var**0.5)
    return xs, means, stds


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "discovery.pdf"
    behavior = dose_response_match(TARGET)
    t0 = time.perf_counter()

    data: dict[str, dict[str, Sequence[float]]] = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in METHODS:
        curves = [curve(stream_for(method, behavior, seed)) for seed in range(SEEDS)]
        xs, means, stds = aggregate(curves)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(xs, lo, hi, color=COLORS[method], alpha=0.18, linewidth=0)
        ax.plot(xs, means, color=COLORS[method], linewidth=2.0, label=f"{LABELS[method]} ({means[-1]:.1f})")
        data[method] = {"budget": xs, "mean": means, "std": stds}
        print(f"  {method:13s} final {means[-1]:.1f} ± {stds[-1]:.1f}")

    ax.set_xlabel("simulation budget")
    ax.set_ylabel("distinct high-scoring topologies")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", title=f"{SEEDS} seeds, mean $\\pm$ std")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    with open(out.rsplit(".", 1)[0] + ".json", "w") as handle:
        json.dump(data, handle, indent=2)
    print(f"\nwrote {out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
