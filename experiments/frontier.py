"""Frontier and sampling diagnostics: the reward-vs-diversity tradeoff as β varies, over seeds.

Two things per β, averaged over seeds. (1) The reward–diversity frontier: mean ± std of distinct
high-scoring topologies (from the tuned eval stream, matching the discovery metric) and mean portfolio
score. (2) The sampling diagnostics on the policy's *topology* distribution: normalized entropy,
top-mode mass, and total-variation distance to the reward-proportional target q ∝ R^β, estimated from a
large policy sample so the empirical topology frequencies are not washed out by the small eval budget.
Because the construction fills a topology's parameters by an environment kernel the policy does not
control, a topology's reward is its mean score over the sampled instances, faithful to what the sampler
marginalizes over; this is what the diagnostics quantify (the sequence→topology gap).

Run: ``uv run python experiments/frontier.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev

from _common import EVAL_BUDGET, MAX_STEPS, PASS, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS

from quiver.eval import evaluate_portfolio, sampling_fidelity
from quiver.generate import tune_score
from quiver.learn.gflownet import sample_terminals, train
from quiver.objective import TargetResponse, dose_response_match, score_once

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = 10
DIAG_SAMPLES = 500  # large policy sample for estimating the topology distribution (diagnostics only)

RESULTS = Path(__file__).parent / "results" / "frontier.json"

Diagnostics = tuple[float | None, float | None, float | None]  # (norm entropy, top-mode mass, TV to R^β)
RunResult = tuple[int, float, Diagnostics]


def run(beta: float, seed: int) -> RunResult:
    behavior = dose_response_match(TARGET)
    policy = train(behavior, steps=TRAIN_STEPS, batch=TRAIN_BATCH, beta=beta, max_steps=MAX_STEPS, seed=seed)
    # Reward–diversity from the tuned eval stream (matches the discovery metric's per-candidate scoring).
    rng = random.Random(100 + seed)
    terminals = sample_terminals(policy, EVAL_BUDGET // (TUNE_STEPS + 1), max_steps=MAX_STEPS, seed=100 + seed)
    stream = [(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS)) for g in terminals]
    scores = [s for _, s in stream]
    report = evaluate_portfolio(scores, [t for t, _ in stream], passed=[s >= PASS for s in scores])
    mean_score = report.quality.mean_score if report.quality and report.passers else 0.0
    # Sampling fidelity from a large nominal-scored sample of the policy's topology distribution.
    diag_terminals = sample_terminals(policy, DIAG_SAMPLES, max_steps=MAX_STEPS, seed=5000 + seed)
    diag = sampling_fidelity(
        [g.topology_hash() for g in diag_terminals],
        [score_once(g, behavior) for g in diag_terminals],
        threshold=PASS,
        beta=beta,
    )
    return report.distinct_passers, mean_score, diag


def _avg(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return mean(present) if present else float("nan")


def main() -> None:
    print(f"{SEEDS} seeds/β | eval {EVAL_BUDGET} sims (tune {TUNE_STEPS}) | train {TRAIN_STEPS * TRAIN_BATCH} sims")
    print(f"diagnostics from {DIAG_SAMPLES} policy samples per (β, seed)\n")
    t0 = time.perf_counter()
    results: dict[float, list[RunResult]] = {beta: [run(beta, s) for s in range(SEEDS)] for beta in BETAS}

    print("reward–diversity frontier (mean ± std over seeds):")
    print(f"{'beta':>5} | {'distinct high-scoring':>22} | {'mean portfolio score':>22}")
    rows: dict[str, dict[str, float]] = {}
    for beta in BETAS:
        d = [distinct for distinct, _, _ in results[beta]]
        q = [score for _, score, _ in results[beta] if score > 0.0] or [0.0]
        print(f"{beta:>5} | {f'{mean(d):.1f} ± {pstdev(d):.1f}':>22} | {f'{mean(q):.3f} ± {pstdev(q):.3f}':>22}")
        rows[str(beta)] = {
            "distinct_mean": mean(d),
            "distinct_std": pstdev(d),
            "score_mean": mean(q),
            "score_std": pstdev(q),
        }

    print("\nsampling diagnostics (mean over seeds; TV to the reward-proportional target q ∝ R^β):")
    print(f"{'beta':>5} | {'norm. entropy':>13} | {'top-mode mass':>13} | {'TV to R^β':>10}")
    for beta in BETAS:
        diags = [diag for _, _, diag in results[beta]]
        ent = _avg([d[0] for d in diags])
        top = _avg([d[1] for d in diags])
        tv = _avg([d[2] for d in diags])
        print(f"{beta:>5} | {ent:>13.3f} | {top:>13.3f} | {tv:>10.3f}")
        rows[str(beta)].update({"norm_entropy": ent, "top_mode_mass": top, "tv_to_reward": tv})

    result = {
        "config": {
            "seeds": SEEDS,
            "eval_budget": EVAL_BUDGET,
            "tune_steps": TUNE_STEPS,
            "train_sims": TRAIN_STEPS * TRAIN_BATCH,
            "diag_samples": DIAG_SAMPLES,
            "betas": list(BETAS),
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "beta": rows,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS}")

    print(f"\ndone in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
