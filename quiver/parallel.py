"""Process-parallel simulation scoring.

Every simulation is independent, so scoring a batch of candidates (across designs, held-out context shocks,
seeds, or targets) is embarrassingly parallel. This module runs those over a process pool. The one
constraint the design turns on: a behavior detector is a closure and does not pickle, so a worker receives a
picklable :class:`SensorCell` (a design plus a :class:`TargetResponse` and optional shared context) and
rebuilds the detector in-process, rather than being handed the closure.

The thread-count environment variables are pinned to one at import so a pool of workers each running an ODE
integration does not oversubscribe the cores; the pool leaves two cores free for the parent and the OS.
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP threading before any numpy-importing module loads (in the parent and in each spawned
# worker, which re-imports this module), so N workers do not each spawn N integration threads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# Silence LOICA's per-assay tqdm progress bars, which otherwise flood a parallel run's output and bury the
# experiment's own progress reporting.
os.environ.setdefault("TQDM_DISABLE", "1")

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Callable, Iterable, TypeVar, cast

from quiver.generate.optimize import tune_score
from quiver.grn import GRN
from quiver.objective import Context, TargetResponse, apply_context, dose_response_match, score_once

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """Worker count for a parallel map: all cores but two, kept for the parent process and the OS."""
    return max(1, cpu_count() - 2)


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["TQDM_DISABLE"] = "1"


def pmap(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
    chunksize: int = 1,
    on_result: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Map ``fn`` over ``items`` across a process pool, preserving input order.

    ``fn`` and every item must be picklable (``fn`` a module-level callable). With ``workers <= 1`` or a
    single item the map runs in-process, so results are identical to a serial ``[fn(x) for x in items]``,
    the pool only changes wall-clock, never the values.

    ``on_result(index, result)`` is invoked in the parent process as each item finishes (in completion
    order, not input order), for live progress and incremental persistence of a long run; the returned list
    is still in input order regardless. Supplying it uses per-item submission (so results stream as they
    complete); without it the map runs in one batch with ``chunksize`` for lower overhead.
    """
    work = list(items)
    if not work:
        return []
    n_workers = default_workers() if workers is None else workers
    if n_workers <= 1 or len(work) == 1:
        results = []
        for i, x in enumerate(work):
            r = fn(x)
            if on_result is not None:
                on_result(i, r)
            results.append(r)
        return results
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        if on_result is None:
            return list(pool.map(fn, work, chunksize=chunksize))
        ordered = cast(list[R], [None] * len(work))
        futures = {pool.submit(fn, x): i for i, x in enumerate(work)}
        for future in as_completed(futures):
            index = futures[future]
            ordered[index] = future.result()
            on_result(index, ordered[index])
        return ordered


@dataclass(frozen=True)
class SensorCell:
    """One picklable unit of sensor scoring: a design against a target, optionally under a shared context.

    ``context`` ``None`` scores the nominal dose-response window; a :class:`Context` applies the shared shock
    (and its assay-window override) exactly as :func:`quiver.objective.score_under_context` does.
    ``tune_steps`` ``0`` is a single simulation; a positive value runs the same local parameter tuning the
    search uses and returns the best score.
    """

    grn: GRN
    target: TargetResponse
    context: Context | None = None
    tune_steps: int = 0
    seed: int = 0


def _run_sensor_cell(cell: SensorCell) -> float:
    behavior = dose_response_match(cell.target)
    grn = apply_context(cell.grn, cell.context) if cell.context is not None else cell.grn
    config = cell.context.config if cell.context is not None else None
    if cell.tune_steps > 0:
        return tune_score(grn, behavior, random.Random(cell.seed), cell.tune_steps, config)
    return score_once(grn, behavior, config)


def score_cells(cells: Iterable[SensorCell], *, workers: int | None = None, chunksize: int = 4) -> list[float]:
    """Score a batch of :class:`SensorCell` in parallel, in input order. Equal to the serial scores."""
    return pmap(_run_sensor_cell, cells, workers=workers, chunksize=chunksize)
