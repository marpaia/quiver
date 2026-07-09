"""Certificate-gated vs. ungated GFlowNet discovery on a sensor target.

Trains two diversity-seeking policies on the same target with everything identical except a monotonicity
reward gate, then measures the discovered portfolio: distinct high-scoring topologies (breadth), the
fraction certified monotone (a well-posed, unique dose-response with a defined EC50), and mean score. The
gate steers discovery toward structurally guaranteed designs; the question this answers empirically is what
that guarantee costs in breadth and quality.

Run: ``uv run python experiments/certificate_gate.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from quiver.generate import tune_score
from quiver.grn import GRN
from quiver.learn.gflownet import sample_terminals, train
from quiver.objective import Behavior, TargetResponse, dose_response_match
from quiver.objective.certificate import is_monotone, monotone_gate

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
SEEDS = 5
EVAL_BUDGET = 300
TUNE_STEPS = 2
PASS = 0.5
TRAIN_STEPS = 40
TRAIN_BATCH = 12
BETA = 4.0
MAX_STEPS = 8
PENALTY = 0.25

RESULTS = Path(__file__).parent / "results" / "certificate_gate.json"

Row = tuple[str, float, bool]


def stream(behavior: Behavior, seed: int, gate: Callable[[GRN], float] | None) -> list[Row]:
    policy = train(behavior, steps=TRAIN_STEPS, batch=TRAIN_BATCH, beta=BETA, max_steps=MAX_STEPS, seed=seed, gate=gate)
    rng = random.Random(1000 + seed)
    terminals = sample_terminals(policy, EVAL_BUDGET // (TUNE_STEPS + 1), max_steps=MAX_STEPS, seed=1000 + seed)
    return [(g.topology_hash(), tune_score(g, behavior, rng, TUNE_STEPS), is_monotone(g)) for g in terminals]


def summarize(rows: list[Row]) -> dict[str, float]:
    passers = [(h, s, m) for h, s, m in rows if s >= PASS]
    distinct = {h for h, _, _ in passers}
    distinct_cert = {h for h, _, m in passers if m}
    n_pass = len(passers)
    cert_pass = sum(1 for _, _, m in passers if m)
    return {
        "distinct": float(len(distinct)),
        "distinct_cert": float(len(distinct_cert)),
        "cert_frac": cert_pass / n_pass if n_pass else 0.0,
        "mean_score": mean([s for _, s, _ in passers]) if passers else 0.0,
        "cert_frac_all": sum(1 for _, _, m in rows if m) / len(rows) if rows else 0.0,
    }


def main() -> None:
    behavior = dose_response_match(TARGET)
    arms: dict[str, Callable[[GRN], float] | None] = {"ungated": None, "gated": monotone_gate(PENALTY)}
    keys = ("distinct", "distinct_cert", "cert_frac", "mean_score", "cert_frac_all")
    agg: dict[str, dict[str, list[float]]] = {a: {k: [] for k in keys} for a in arms}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        for arm, gate in arms.items():
            s = summarize(stream(behavior, seed, gate))
            for k in keys:
                agg[arm][k].append(s[k])
        row = " | ".join(
            f"{arm} distinct={agg[arm]['distinct'][-1]:.0f} cert%={agg[arm]['cert_frac'][-1]:.0%}" for arm in arms
        )
        print(f"  seed {seed}: {row}")

    print(f"\nran {SEEDS} seeds x {len(arms)} arms in {time.perf_counter() - t0:.0f}s\n")
    summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        d = agg[arm]
        print(
            f"  {arm:8s} distinct {mean(d['distinct']):.1f} ± {pstdev(d['distinct']):.1f} | "
            f"certified-distinct {mean(d['distinct_cert']):.1f} | "
            f"cert-frac of passers {mean(d['cert_frac']):.0%} | "
            f"cert-frac all sampled {mean(d['cert_frac_all']):.0%} | "
            f"mean score {mean(d['mean_score']):.3f}"
        )
        summary[arm] = {
            "distinct_mean": mean(d["distinct"]),
            "distinct_std": pstdev(d["distinct"]),
            "certified_distinct_mean": mean(d["distinct_cert"]),
            "cert_frac_of_passers": mean(d["cert_frac"]),
            "cert_frac_all_sampled": mean(d["cert_frac_all"]),
            "mean_score": mean(d["mean_score"]),
        }

    result = {
        "config": {
            "seeds": SEEDS,
            "eval_budget": EVAL_BUDGET,
            "tune_steps": TUNE_STEPS,
            "train_sims": TRAIN_STEPS * TRAIN_BATCH,
            "beta": BETA,
            "penalty": PENALTY,
            "target": {"ec50": TARGET.ec50, "dynamic_range": TARGET.dynamic_range, "direction": TARGET.direction},
        },
        "arms": summary,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
