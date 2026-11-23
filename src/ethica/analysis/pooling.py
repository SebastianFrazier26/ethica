"""Contrasts pooled over training runs (study v1).

In v0 an arm was one model and the only noise was sampling. In v1 an arm is a training recipe,
and one adapter per arm cannot separate "this framework" from "this training run". Each arm is
therefore trained with several seeds, each training run evaluated in its own experiment, and
the arm-level contrast is computed here with the training run as a unit that is resampled.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Cell = Mapping[tuple[str, int], float]  # (task id, sampling seed) -> score
FloatArray = npt.NDArray[np.float64]

# which scorer's rows represent a metric when several scored it (as in the report)
_SCORER_RANK = ("judge:", "programmatic", "exact", "heuristic:")


@dataclass(frozen=True)
class PooledContrast:
    estimate: float  # mean over arm runs minus mean over reference runs
    lo: float
    hi: float
    arm_run_means: dict[str, float]
    ref_run_means: dict[str, float]
    n_tasks: int
    # False when either arm has a single training run: the interval then reflects tasks only
    # and says nothing about run-to-run variation
    runs_resampled: bool
    # every arm run lies on the same side of every reference run
    sign_consistent: bool


def cells_by_arm(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, dict[tuple[str, int], float]]:
    """arm ("<model>|<framework>|<method>") -> cell, from `Store.scores_table` rows, using one
    scorer family for the metric: judge if present, else programmatic, exact, heuristic."""
    mine = [r for r in rows if r["metric"] == metric and not math.isnan(r["value"])]
    family = next((f for f in _SCORER_RANK if any(r["scorer"].startswith(f) for r in mine)), None)
    acc: dict[str, dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in mine:
        if family is None or r["scorer"].startswith(family):
            arm = f"{r['model_key']}|{r['framework']}|{r['method']}"
            acc[arm][(r["task_id"], r["seed"])].append(r["value"])
    return {a: {k: float(np.mean(v)) for k, v in c.items()} for a, c in acc.items()}


def _task_means(cell: Cell) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for (task, _), v in cell.items():
        if not math.isnan(v):
            acc[task].append(v)
    return {t: float(np.mean(v)) for t, v in acc.items()}


def _matrix(runs: Mapping[str, Cell], tasks: Sequence[str]) -> FloatArray:
    means = [_task_means(c) for c in runs.values()]
    return np.array([[m[t] for t in tasks] for m in means], dtype=np.float64)


def pooled_contrast(
    arm: Mapping[str, Cell],
    ref: Mapping[str, Cell],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PooledContrast:
    """`arm` and `ref` map a training-run id to that run's cell. Tasks are paired (the same
    tasks are resampled for both arms); training runs are not (run 0 of one arm has nothing to
    do with run 0 of the other), so each arm's runs are resampled independently.

    With few runs the bootstrap understates run-to-run variation; read the per-run means and
    `sign_consistent` next to the interval."""
    if not arm or not ref:
        raise ValueError("need at least one training run per arm")
    task_sets = [set(_task_means(c)) for c in (*arm.values(), *ref.values())]
    tasks = sorted(set.intersection(*task_sets))
    if not tasks:
        raise ValueError("no task is scored in every training run of both arms")
    a, r = _matrix(arm, tasks), _matrix(ref, tasks)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ti = rng.integers(0, len(tasks), len(tasks))
        ai = rng.integers(0, len(a), len(a))
        ri = rng.integers(0, len(r), len(r))
        stats[b] = a[ai][:, ti].mean() - r[ri][:, ti].mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    arm_means = {k: float(m) for k, m in zip(arm, a.mean(axis=1), strict=True)}
    ref_means = {k: float(m) for k, m in zip(ref, r.mean(axis=1), strict=True)}
    av, rv = list(arm_means.values()), list(ref_means.values())
    return PooledContrast(
        estimate=float(a.mean() - r.mean()),
        lo=float(lo),
        hi=float(hi),
        arm_run_means=arm_means,
        ref_run_means=ref_means,
        n_tasks=len(tasks),
        runs_resampled=len(a) > 1 and len(r) > 1,
        sign_consistent=max(av) < min(rv) or min(av) > max(rv),
    )
