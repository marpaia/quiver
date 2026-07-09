"""Does a structure-controlling policy trade breadth for control of the topology distribution?

The env-random policy (`learn.gflownet`) leaves each move's regulator choice to the environment kernel, so
its topology distribution stays near-maximum-entropy at every reward exponent (the frontier diagnostics):
tempering shapes the sequence distribution but not the topology marginal, which is what delivers a broad,
near-uniform portfolio over the high-scoring set. The structure-controlling policy (`learn.structured`)
chooses those regulators itself, so the policy shapes the topology marginal directly and can concentrate it
as beta grows. Neither construction is injective onto topologies (`experiments/injectivity.py`), so both
sample topologies in proportion to reward only up to a path-multiplicity factor.

This trains both policies over a beta sweep and reports, per (variant, beta) and averaged over seeds: the
sampling diagnostics on the topology distribution (normalized entropy, top-mode mass, and TV to the
reward-proportional target) from a large policy sample, and the delivered portfolio (distinct high-scoring
topologies and mean portfolio score at the eval budget). Together these show the axis: the env-random policy
maximizes breadth at every beta, the structure-controlling policy converts beta into concentration at the
cost of breadth.

Run: ``uv run python experiments/structured_diagnostics.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev

from _common import env_int

from quiver.eval import evaluate_portfolio, sampling_fidelity
from quiver.generate import tune_score
from quiver.learn import gflownet, structured
from quiver.objective import TargetResponse, dose_response_match, score_once
from quiver.parallel import pmap

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = env_int("SEEDS", 10)
EVAL_BUDGET = 300
DIAG_SAMPLES = 500
TUNE_STEPS = 2
PASS = 0.5
# Both variants share a training budget long enough for the structure-controlling policy to shape the
# topology distribution; at a short budget it under-concentrates and beta has little visible effect.
TRAIN_STEPS = 200
TRAIN_BATCH = 12
MAX_STEPS = 8
VARIANTS = {"env-random": gflownet, "structured": structured}
RESULTS = Path(__file__).parent / "results" / "structured_diagnostics.json"

RunKey = tuple[str, float, int]  # (variant, beta, seed)
Diagnostics = tuple[float | None, float | None, float | None]  # (norm entropy, top-mode mass, TV to R^beta)
RunResult = tuple[int, float, Diagnostics]  # (distinct high-scoring, mean portfolio score, diagnostics)


def run(key: RunKey) -> RunResult:
    """Train one variant at one (beta, seed) and return its delivered breadth, mean portfolio score, and
    topology-distribution diagnostics. Picklable and self-contained, so a process pool runs the sweep."""
    variant, beta, seed = key
    module = VARIANTS[variant]
    behavior = dose_response_match(TARGET)
    policy = module.train(behavior, steps=TRAIN_STEPS, batch=TRAIN_BATCH, beta=beta, max_steps=MAX_STEPS, seed=seed)

    # Delivered portfolio from the tuned eval stream (matches the discovery metric's per-candidate scoring).
    n_eval = EVAL_BUDGET // (TUNE_STEPS + 1)
    eval_terminals = module.sample_terminals(policy, n_eval, max_steps=MAX_STEPS, seed=100 + seed)
    rng = random.Random(100 + seed)
    stream = [(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS)) for g in eval_terminals]
    scores = [s for _, s in stream]
    report = evaluate_portfolio(scores, [h for h, _ in stream], passed=[s >= PASS for s in scores])
    mean_score = report.quality.mean_score if report.quality and report.passers else 0.0

    # Topology-distribution diagnostics from a large nominal-scored policy sample.
    diag_terminals = module.sample_terminals(policy, DIAG_SAMPLES, max_steps=MAX_STEPS, seed=5000 + seed)
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


def _summarize(runs: list[RunResult]) -> dict[str, float | list[int]]:
    distinct = [d for d, _, _ in runs]
    score = [s for _, s, _ in runs]
    diags = [g for _, _, g in runs]
    return {
        "distinct_mean": mean(distinct),
        "distinct_std": pstdev(distinct),
        "score_mean": mean(score),
        "score_std": pstdev(score),
        "entropy": _avg([g[0] for g in diags]),
        "top_mode_mass": _avg([g[1] for g in diags]),
        "tv_to_reward_proportional": _avg([g[2] for g in diags]),
        "distinct_raw": distinct,
    }


def main() -> None:
    keys: list[RunKey] = [(v, b, s) for v in VARIANTS for b in BETAS for s in range(SEEDS)]
    print(
        f"{SEEDS} seeds/beta | betas {BETAS} | {len(keys)} runs | eval {EVAL_BUDGET} sims "
        f"| diagnostics {DIAG_SAMPLES} samples | train {TRAIN_STEPS * TRAIN_BATCH} sims/run",
        flush=True,
    )
    t0 = time.perf_counter()
    done = [0]

    def progress(_index: int, _result: RunResult) -> None:
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(keys):
            print(f"  {done[0]}/{len(keys)} runs done ({time.perf_counter() - t0:.0f}s)", flush=True)

    flat = pmap(run, keys, on_result=progress)
    runs: dict[str, dict[float, list[RunResult]]] = {v: {b: [] for b in BETAS} for v in VARIANTS}
    for key, result in zip(keys, flat):
        variant, beta, _ = key
        runs[variant][beta].append(result)

    summary = {v: {str(b): _summarize(runs[v][b]) for b in BETAS} for v in VARIANTS}
    RESULTS.write_text(
        json.dumps(
            {
                "config": {
                    "seeds": SEEDS,
                    "betas": list(BETAS),
                    "train_steps": TRAIN_STEPS,
                    "train_batch": TRAIN_BATCH,
                    "eval_budget": EVAL_BUDGET,
                    "diag_samples": DIAG_SAMPLES,
                    "max_steps": MAX_STEPS,
                },
                "variants": summary,
            },
            indent=2,
        )
        + "\n"
    )

    print("\nsampling diagnostics on the topology distribution (mean over seeds):")
    print(f"  {'variant':12s} {'beta':>5} | {'entropy':>8} {'top-mode':>9} {'TV to R^b':>10}")
    for variant in VARIANTS:
        for beta in BETAS:
            s = summary[variant][str(beta)]
            print(
                f"  {variant:12s} {beta:>5} | {s['entropy']:>8.3f} {s['top_mode_mass']:>9.3f} "
                f"{s['tv_to_reward_proportional']:>10.3f}"
            )

    print("\ndelivered portfolio (mean over seeds):")
    print(f"  {'variant':12s} {'beta':>5} | {'distinct high-scoring':>22} | {'mean portfolio score':>20}")
    for variant in VARIANTS:
        for beta in BETAS:
            s = summary[variant][str(beta)]
            distinct_cell = f"{s['distinct_mean']:.1f} ± {s['distinct_std']:.1f}"
            score_cell = f"{s['score_mean']:.3f} ± {s['score_std']:.3f}"
            print(f"  {variant:12s} {beta:>5} | {distinct_cell:>22} | {score_cell:>20}")

    print(f"\nwrote {RESULTS}\ndone in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
