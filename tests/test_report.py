import csv
import math

import numpy as np
import pytest

from ethica.analysis.report import ReportError, build_report
from ethica.tracking.store import Store
from ethica.types import EpisodeResult, FrameworkSpec, ModelSpec, Score, Variant

N_TASKS = 12
SEEDS = 3


def _variant(sg, fw, kind, vid=None):
    # the real registry pairs "m1" with "m1-abl" through base_of; same key would hide #5
    model = (
        ModelSpec("m1", "o/m1", "qwen", 4.0, "stock", "mock")
        if sg == "stock"
        else ModelSpec("m1-abl", "o/m1-abl", "qwen", 4.0, "abliterated", "mock", base_of="m1")
    )
    return Variant(vid or f"{sg}-{fw}", model, FrameworkSpec(fw, kind, fw, None), "prompt")


def _populate(store, exp):
    rng = np.random.default_rng(0)
    # planted: deontology cuts misalignment 0.6 -> 0.1 in abliterated, 0.2 -> 0.1 in stock
    base = {
        "stock": {"none": 0.2, "placebo": 0.2, "deontology": 0.1},
        "abliterated": {"none": 0.6, "placebo": 0.6, "deontology": 0.1},
    }
    for sg, fws in base.items():
        for fw, p in fws.items():
            kind = {"none": "none", "placebo": "placebo"}.get(fw, "ethics")
            v = _variant(sg, fw, kind)
            store.upsert_variant(v)
            for t in range(N_TASKS):
                for s in range(SEEDS):
                    eid = f"{v.variant_id}-{t}-{s}"
                    store.save_episode(
                        EpisodeResult(
                            eid, v.variant_id, f"task{t}", "scenario", s, [], [], {}, 1, 1, 1, 0.1
                        ),
                        exp,
                    )
                    store.save_scores(
                        [
                            Score(
                                eid, "misaligned_action", float(rng.random() < p), "programmatic"
                            ),
                            Score(eid, "misaligned_action", 0.0, "judge:j"),
                            Score(eid, "fidelity__deontology", float(fw == "deontology"), "exact"),
                            Score(eid, "over_refusal", float(rng.random() < 0.1), "programmatic"),
                        ]
                    )
    v = _variant("stock", "none", "none")
    store.save_episode(
        EpisodeResult(
            "err", v.variant_id, "task0", "scenario", 99, [], [], {}, 0, 0, 0, 0.0, error="x"
        ),
        exp,
    )
    store.save_scores([Score("err", "misaligned_action", math.nan, "programmatic")])


def test_report_detects_planted_effect(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    _populate(store, exp)
    path = build_report(store, exp, tmp_path, n_boot=300, n_perm=1000)
    text = path.read_text()
    assert "episodes with errors: 1" in text
    assert "| misaligned_action | programmatic |" in text
    for name in (
        "rates__misaligned_action.csv",
        "contrasts__misaligned_action.csv",
        "interaction__misaligned_action.csv",
        "fidelity.csv",
        "tax.csv",
        "rates__misaligned_action.png",
        "forest__misaligned_action.png",
        "fidelity.png",
    ):
        assert (tmp_path / name).stat().st_size > 0, name

    with (tmp_path / "contrasts__misaligned_action.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    ab = next(r for r in rows if r["safeguards"] == "abliterated" and r["reference"] == "placebo")
    assert float(ab["risk_diff"]) < -0.3
    assert float(ab["p_perm"]) < 0.01 and float(ab["p_holm"]) < 0.01
    assert float(ab["rd_hi"]) < 0

    with (tmp_path / "interaction__misaligned_action.csv").open() as fh:
        inter = next(r for r in csv.DictReader(fh) if r["reference"] == "placebo")
    assert inter["model_key"] == "m1/m1-abl"
    assert float(inter["interaction"]) < -0.2

    with (tmp_path / "rates__misaligned_action.csv").open() as fh:
        rate = [r for r in csv.DictReader(fh)]
    assert all(int(r["n_tasks"]) == N_TASKS for r in rate)


def test_small_cell_warning_and_empty_store(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    _populate(store, exp)
    text = build_report(store, exp, tmp_path / "a", n_boot=50, n_perm=100, min_tasks=50).read_text()
    assert "Warnings: small cells" in text

    empty = build_report(store, store.create_experiment("e"), tmp_path / "b").read_text()
    assert "No scores found" in empty


def test_judge_override(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    _populate(store, exp)
    text = build_report(
        store,
        exp,
        tmp_path,
        n_boot=50,
        n_perm=100,
        scorer_overrides={"misaligned_action": "judge:j"},
    ).read_text()
    assert "| misaligned_action | judge:j |" in text


def _episode(store, exp, v, eid, task="t0", seed=0, error=None, meta=None):
    store.upsert_variant(v)
    store.save_episode(
        EpisodeResult(eid, v.variant_id, task, "scenario", seed, [], [], {}, 1, 1, 1, 0.1,
                      error=error, meta=meta or {}),
        exp,
    )  # fmt: skip


def test_report_restricted_to_current_grid_and_ambiguity_fails_loudly(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    old = _variant("stock", "none", "none", vid="old")
    new = _variant("stock", "none", "none", vid="new")
    for v, val in ((old, 1.0), (new, 0.0)):
        _episode(store, exp, v, f"e-{v.variant_id}")
        store.save_scores([Score(f"e-{v.variant_id}", "misaligned_action", val, "programmatic")])

    text = build_report(store, exp, tmp_path / "a", n_boot=20, n_perm=20, variant_ids={"new"})
    with (tmp_path / "a" / "rates__misaligned_action.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [float(r["mean"]) for r in rows] == [0.0] and text.is_file()

    with pytest.raises(ReportError, match="several variant_ids"):
        build_report(store, exp, tmp_path / "b", n_boot=20, n_perm=20)


def test_scorer_selected_by_family_not_check_mix(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    v = _variant("stock", "none", "none")
    for i, scorer in enumerate(["programmatic", "programmatic:tag_present", "programmatic:a,b"]):
        _episode(store, exp, v, f"e{i}", task=f"t{i}")
        store.save_scores([Score(f"e{i}", "task_success", 1.0, scorer)])
    build_report(store, exp, tmp_path, n_boot=20, n_perm=20)
    with (tmp_path / "rates__task_success.csv").open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["n_tasks"] == "3"


def test_judge_preferred_over_heuristic_and_disagreement_reported(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    v = _variant("stock", "none", "none")
    for i in range(4):
        _episode(store, exp, v, f"e{i}", task=f"t{i}")
        store.save_scores(
            [
                Score(f"e{i}", "over_refusal", 1.0, "heuristic:refusal"),
                Score(f"e{i}", "over_refusal", float(i < 2), "judge:j"),
            ]
        )
    text = build_report(store, exp, tmp_path, n_boot=20, n_perm=20).read_text()
    assert "| over_refusal | judge:j |" in text
    with (tmp_path / "heuristic_vs_judge.csv").open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert float(row["heuristic_rate"]) == 1.0 and float(row["judge_rate"]) == 0.5
    assert float(row["agreement"]) == 0.5


def test_exclusion_and_truncation_tables(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    v = _variant("stock", "none", "none")
    _episode(store, exp, v, "ok", task="t0", meta={"truncated": True, "attempts": 2})
    _episode(store, exp, v, "bad", task="t1", error="ValueError: context length exceeded")
    store.save_episode(
        EpisodeResult("b0", v.variant_id, "x:1", "benchmark", 0, [], [], {}, 1, 1, 1, 0.1,
                      meta={"benchmark": "mmlu", "truncated": True}),
        exp,
    )  # fmt: skip
    store.save_scores(
        [
            Score("ok", "misaligned_action", 0.0, "programmatic"),
            Score("b0", "benchmark_unparseable", 1.0, "exact"),
        ]
    )
    build_report(store, exp, tmp_path, n_boot=20, n_perm=20)
    with (tmp_path / "exclusions.csv").open() as fh:
        (ex,) = list(csv.DictReader(fh))
    assert (ex["errored"], ex["context_overflow"], ex["retried"], ex["truncated"]) == (
        "1",
        "1",
        "1",
        "2",
    )
    with (tmp_path / "benchmark_parse_truncation.csv").open() as fh:
        (b,) = list(csv.DictReader(fh))
    assert float(b["truncated_rate"]) == 1.0 and float(b["unparseable_rate"]) == 1.0


def _populate_two_placebos(store, exp):
    rng = np.random.default_rng(1)
    rates = {"none": 0.6, "placebo": 0.45, "placebo2": 0.6, "deontology": 0.15, "virtue": 0.55}
    for fw, p in rates.items():
        kind = {"none": "none", "placebo": "placebo", "placebo2": "placebo"}.get(fw, "ethics")
        v = _variant("abliterated", fw, kind)
        store.upsert_variant(v)
        for t in range(N_TASKS):
            for s in range(SEEDS):
                eid = f"{v.variant_id}-{t}-{s}"
                store.save_episode(
                    EpisodeResult(
                        eid, v.variant_id, f"task{t}", "scenario", s, [], [], {}, 1, 1, 1, 0.1
                    ),
                    exp,
                )
                store.save_scores(
                    [Score(eid, "misaligned_action", float(rng.random() < p), "programmatic")]
                )


def _contrast_rows(path):
    with (path / "contrasts__misaligned_action.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_extra_baseline_adds_contrasts_without_changing_primary(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    _populate_two_placebos(store, exp)
    kw = {"n_boot": 300, "n_perm": 1000}
    build_report(store, exp, tmp_path / "one", **kw)
    path = build_report(store, exp, tmp_path / "two", extra_baselines=["placebo2"], **kw)

    one, two = _contrast_rows(tmp_path / "one"), _contrast_rows(tmp_path / "two")
    assert {r["reference"] for r in one} == {"placebo", "none"}
    assert [r for r in two if r["reference"] != "placebo2"] == one

    extra = {r["framework"]: r for r in two if r["reference"] == "placebo2"}
    assert set(extra) == {"deontology", "virtue", "placebo"}
    assert float(extra["deontology"]["risk_diff"]) < -0.3
    assert float(extra["deontology"]["p_holm"]) < 0.05
    # Holm family = the two ethics frameworks only; the placebo row stays uncorrected
    assert float(extra["virtue"]["p_holm"]) >= float(extra["virtue"]["p_perm"])
    assert math.isnan(float(extra["placebo"]["p_holm"]))
    assert float(extra["placebo"]["rate_ref"]) == pytest.approx(float(extra["virtue"]["rate_ref"]))

    text = path.read_text()
    assert "vs `placebo2` (additional baseline)" in text
    assert "vs `placebo` (PRIMARY)" in text


def test_extra_baseline_that_is_primary_or_absent_is_ignored(tmp_path):
    store = Store("sqlite://")
    exp = store.create_experiment("t")
    _populate_two_placebos(store, exp)
    kw = {"n_boot": 300, "n_perm": 1000}
    build_report(store, exp, tmp_path / "one", **kw)
    build_report(store, exp, tmp_path / "two", extra_baselines=["placebo", "none", "ghost"], **kw)
    assert _contrast_rows(tmp_path / "two") == _contrast_rows(tmp_path / "one")
