import math

import numpy as np
import pytest

from ethica.analysis.pooling import cells_by_arm, pooled_contrast
from ethica.analysis.stats import mean_ci_cluster_bootstrap

TASKS = [f"t{i}" for i in range(40)]


def _cell(rate, seed, sampling_seeds=(0, 1)):
    rng = np.random.default_rng(seed)
    return {(t, s): float(rng.random() < rate) for t in TASKS for s in sampling_seeds}


def _runs(rates, base_seed):
    return {f"s{i}": _cell(r, base_seed + i) for i, r in enumerate(rates)}


def test_detects_an_effect_shared_by_every_run():
    c = pooled_contrast(_runs([0.2, 0.25, 0.2], 10), _runs([0.6, 0.55, 0.6], 20), n_boot=500)
    assert c.estimate < -0.25 and c.hi < 0
    assert c.runs_resampled and c.sign_consistent and c.n_tasks == 40
    assert set(c.arm_run_means) == {"s0", "s1", "s2"}
    assert c.estimate == pytest.approx(
        np.mean(list(c.arm_run_means.values())) - np.mean(list(c.ref_run_means.values()))
    )


def test_one_lucky_run_does_not_make_an_arm_effect():
    # same true rate everywhere except one arm run that came out low
    arm, ref = _runs([0.1, 0.5, 0.5], 30), _runs([0.5, 0.5, 0.5], 40)
    pooled = pooled_contrast(arm, ref, n_boot=800)
    assert pooled.lo < 0 < pooled.hi and not pooled.sign_consistent
    # the lucky run alone, analysed as v0 would, looks like a clear effect
    alone = pooled_contrast({"s0": arm["s0"]}, {"s0": ref["s0"]}, n_boot=800)
    assert alone.hi < 0 and not alone.runs_resampled


def test_single_runs_reduce_to_the_task_clustered_paired_difference():
    a, r = _cell(0.3, 1), _cell(0.6, 2)
    c = pooled_contrast({"x": a}, {"y": r}, n_boot=1500)
    keys = sorted(a)
    diffs = [a[k] - r[k] for k in keys]
    ci = mean_ci_cluster_bootstrap(diffs, [k[0] for k in keys], n_boot=1500)
    assert c.estimate == pytest.approx(ci.mean)
    assert c.lo == pytest.approx(ci.lo, abs=0.04) and c.hi == pytest.approx(ci.hi, abs=0.04)


def test_only_tasks_scored_in_every_run_are_used():
    a = _cell(0.3, 1)
    r = {k: v for k, v in _cell(0.6, 2).items() if k[0] not in ("t0", "t1")}
    r[("t2", 0)] = math.nan
    r[("t2", 1)] = math.nan
    assert pooled_contrast({"x": a}, {"y": r}, n_boot=50).n_tasks == 37
    with pytest.raises(ValueError, match="no task is scored"):
        pooled_contrast({"x": {("only_a", 0): 1.0}}, {"y": {("only_b", 0): 1.0}})
    with pytest.raises(ValueError, match="at least one training run"):
        pooled_contrast({}, {"y": r})


def test_deterministic():
    arm, ref = _runs([0.2, 0.3], 1), _runs([0.5, 0.6], 5)
    assert pooled_contrast(arm, ref, n_boot=200) == pooled_contrast(arm, ref, n_boot=200)


def _row(framework, task, value, scorer, metric="harmful_compliance", seed=0):
    return {
        "model_key": "m", "framework": framework, "method": "none", "task_id": task,
        "seed": seed, "metric": metric, "value": value, "scorer": scorer,
    }  # fmt: skip


def test_cells_by_arm_uses_one_scorer_family():
    rows = [
        _row("virtue", "t1", 1.0, "heuristic:refusal"),
        _row("virtue", "t1", 0.0, "judge:j"),
        _row("virtue", "t2", 1.0, "heuristic:refusal"),  # judge did not score t2
        _row("generic", "t1", 1.0, "judge:j"),
        _row("generic", "t1", math.nan, "judge:j", seed=1),
        _row("generic", "t1", 1.0, "judge:j", metric="over_refusal"),
    ]
    cells = cells_by_arm(rows, "harmful_compliance")
    assert cells == {"m|virtue|none": {("t1", 0): 0.0}, "m|generic|none": {("t1", 0): 1.0}}
    assert cells_by_arm(rows[:1], "harmful_compliance") == {"m|virtue|none": {("t1", 0): 1.0}}
    assert cells_by_arm(rows, "task_success") == {}
