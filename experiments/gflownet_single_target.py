"""Is the GFlowNet's advantage real across seeds?

Runs random, evolutionary, and GFlowNet search over the sensor topology space across several seeds and
reports distinct high-scoring topologies as mean ± std. A single run is an anecdote; non-overlapping
spreads across seeds are what make "the GFlowNet discovers a broader portfolio" a defensible claim.

Run: ``uv run python experiments/gflownet_single_target.py`` (needs the ``sim`` and ``learn`` extras).
"""

from __future__ import annotations

import time
from statistics import mean, pstdev

from _common import BETA, EVAL_BUDGET, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS, portfolio
from _gfn import gflownet_stream

from quiver.eval import paired_summary
from quiver.generate import evolutionary_search, random_search
from quiver.objective import TargetResponse, dose_response_match

TARGET = TargetResponse("GFP", ec50=1.0, dynamic_range=100.0, direction="any")
SEEDS = 10
METHODS = ("random", "evolutionary", "gflownet")


def main() -> None:
    behavior = dose_response_match(TARGET)
    train_sims = TRAIN_STEPS * TRAIN_BATCH
    print(
        f"{SEEDS} seeds | eval {EVAL_BUDGET} sims/method (tune {TUNE_STEPS}) | GFN train {train_sims} sims, β={BETA}\n"
    )

    final: dict[str, list[int]] = {m: [] for m in METHODS}
    midway: dict[str, list[int]] = {m: [] for m in METHODS}
    t0 = time.perf_counter()
    for seed in range(SEEDS):
        streams = {
            "random": random_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed),
            "evolutionary": evolutionary_search(behavior, budget=EVAL_BUDGET, tune_steps=TUNE_STEPS, seed=seed),
            "gflownet": gflownet_stream(behavior, seed),
        }
        for name in METHODS:
            distinct, mid = portfolio(streams[name])
            final[name].append(distinct)
            midway[name].append(mid)
        row = " | ".join(f"{name} {final[name][-1]}" for name in METHODS)
        print(f"  seed {seed}: {row}")

    print(f"\nran {SEEDS} seeds in {time.perf_counter() - t0:.0f}s\n")
    print("distinct high-scoring topologies (mean ± std over seeds):")
    for name in METHODS:
        f_m, f_s = mean(final[name]), pstdev(final[name])
        m_m, m_s = mean(midway[name]), pstdev(midway[name])
        print(f"  {name:13s} final {f_m:.1f} ± {f_s:.1f}   |   @50% budget {m_m:.1f} ± {m_s:.1f}")

    print("\n== verdict ==")
    ordered = sorted(METHODS, key=lambda m: mean(final[m]), reverse=True)
    summary = ", ".join(f"{name} {mean(final[name]):.1f}" for name in ordered)
    print(f"  distinct designs (mean over {SEEDS} seeds): {summary}")

    print("\n== paired advantage over seeds (cancels seed difficulty) ==")
    for opponent in ("evolutionary", "random"):
        s = paired_summary(final["gflownet"], final[opponent])
        print(
            f"  gfn − {opponent:12s} +{s.mean_diff:.2f} [95% CI {s.ci_low:+.2f}, {s.ci_high:+.2f}] "
            f"| win {s.win_rate:.0%} (strict {s.strict_win_rate:.0%}) | sign-test p={s.sign_p:.1e} | n={s.n}"
        )


if __name__ == "__main__":
    main()
