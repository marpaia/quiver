"""The experiment protocol's binding of the library GFlowNet discovery stream (needs the ``learn`` extra).

:func:`quiver.learn.gflownet.discovery_stream` is the reusable train-then-sample-then-score pipeline;
this binds it to the shared experiment knobs (:mod:`_common`) so the single-target, quality-diversity, and
discovery-figure runs configure the sampler identically. The seed offset separates the sampling and tuning
RNG from the training seed, matching how the classical search baselines seed each run.
"""

from __future__ import annotations

from typing import Callable

from _common import BETA, EVAL_BUDGET, MAX_STEPS, TRAIN_BATCH, TRAIN_STEPS, TUNE_STEPS, Stream

from quiver.grn import GRN
from quiver.learn.gflownet import discovery_stream
from quiver.objective import Behavior


def gflownet_stream(
    behavior: Behavior,
    seed: int,
    *,
    beta: float = BETA,
    gate: Callable[[GRN], float] | None = None,
    sample_offset: int = 1000,
    n: int | None = None,
) -> Stream:
    """A discovery stream from a GFlowNet trained under the shared experiment knobs (train steps, batch,
    reward exponent, eval budget). ``sample_offset`` offsets the sampling and tuning seed from the training
    seed; ``gate`` optionally reshapes the training reward."""
    return discovery_stream(
        behavior,
        seed=seed,
        beta=beta,
        steps=TRAIN_STEPS,
        batch=TRAIN_BATCH,
        max_steps=MAX_STEPS,
        n_samples=n if n is not None else EVAL_BUDGET // (TUNE_STEPS + 1),
        tune_steps=TUNE_STEPS,
        sample_seed=sample_offset + seed,
        gate=gate,
    )
