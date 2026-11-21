import json
from pathlib import Path

import pytest
from orch_helpers import Tracker, make_cfg, script
from rich.console import Console

from ethica import orchestrator as orch
from ethica.backends.mock import MockBackend
from ethica.config import ConfigError, load_config, validate_refs
from ethica.tracking.store import Store
from ethica.types import Score

QUIET = Console(quiet=True)


def run_all(cfg, tracker=None):
    tracker = tracker or Tracker()
    stats = orch.generate(cfg, backend_factory=tracker, console=QUIET)
    orch.score(cfg, console=QUIET)
    return stats, tracker


def test_end_to_end_generate_score_report(tmp_path):
    cfg = make_cfg(tmp_path, methods=["prompt", "rag"])
    stats, _ = run_all(cfg)
    # 6 variants x (1 scenario + 3 fidelity items)
    assert stats.done == 24 and stats.errors == 0 and stats.skipped == 0
    store = Store(cfg.resolved_db_url)
    try:
        rows = store.scores_table(cfg.name)
        ts = [r for r in rows if r["metric"] == "task_success" and r["task_kind"] == "scenario"]
        assert len(ts) == 6 and all(r["value"] == 1.0 for r in ts)
        assert {r["metric"] for r in rows} >= {"parse_failure_rate", "fidelity__deontology"}
    finally:
        store.close()
    path = orch.report(cfg, console=QUIET)
    assert path.is_file() and "misaligned_action" in path.read_text()


def test_rag_index_built_once_and_variant_ids_unique(tmp_path):
    cfg = make_cfg(tmp_path, methods=["prompt", "rag"])
    variants = orch.expand_grid(cfg)
    assert len({v.variant_id for v in variants}) == len(variants) == 6
    run_all(cfg)
    assert list((Path(cfg.out_dir) / "rag").glob("*.json"))


def test_resume_skips_completed_and_runs_only_new(tmp_path):
    cfg = make_cfg(tmp_path)
    first, _ = run_all(cfg)
    again, tracker = run_all(cfg)
    assert again.done == 0 and again.skipped == first.done
    assert tracker.loads == []  # fully done -> no model is even loaded

    wider = make_cfg(tmp_path, seeds=[0, 1])
    third, _ = run_all(wider)
    assert third.done == first.done and third.skipped == first.done


def test_scoring_is_idempotent(tmp_path):
    cfg = make_cfg(tmp_path)
    run_all(cfg)

    def count() -> int:
        s = Store(cfg.resolved_db_url)
        try:
            return len(s.scores_table(cfg.name))
        finally:
            s.close()

    n = count()
    orch.score(cfg, console=QUIET)
    assert count() == n


def test_one_model_resident_at_a_time(tmp_path):
    cfg = make_cfg(tmp_path, models=["mock-a", "mock-b"])
    _, tracker = run_all(cfg)
    assert tracker.max_open == 1 and tracker.open == 0
    assert [k for k, _ in tracker.loads] == ["mock-a", "mock-b"]


def test_episode_exception_recorded_and_batch_continues(tmp_path, monkeypatch):
    real = orch.run_episode
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("kaboom")
        return real(*a, **kw)

    monkeypatch.setattr(orch, "run_episode", flaky)
    cfg = make_cfg(tmp_path)
    stats, _ = run_all(cfg)
    assert stats.errors == 1 and stats.done == 12
    store = Store(cfg.resolved_db_url)
    try:
        errs = [e for e in store.iter_episodes(cfg.name) if e.error]
        assert len(errs) == 1 and "kaboom" in (errs[0].error or "")
    finally:
        store.close()


def test_backend_error_inside_episode_is_stored(tmp_path):
    def boom(spec, adapter=None):
        return MockBackend(lambda m: (_ for _ in ()).throw(RuntimeError("gpu on fire")))

    cfg = make_cfg(tmp_path, frameworks=["none"])
    stats = orch.generate(cfg, backend_factory=boom, console=QUIET)
    assert stats.errors == 4 and stats.done == 4


def test_model_load_failure_skips_group_not_run(tmp_path):
    def factory(spec, adapter=None):
        if spec.key == "mock-a":
            raise OSError("no weights")
        return MockBackend(script, model_id=spec.key)

    cfg = make_cfg(tmp_path, models=["mock-a", "mock-b"], frameworks=["none"])
    stats = orch.generate(cfg, backend_factory=factory, console=QUIET)
    assert stats.done == 4 and len(stats.failed_variants) == 1


def test_offline_first_falls_back_online():
    seen = []

    def fn():
        import os

        seen.append(os.environ.get("HF_HUB_OFFLINE"))
        if len(seen) == 1:
            raise OSError("not cached")
        return "ok"

    assert orch.offline_first(fn, "m", QUIET) == "ok"
    assert seen == ["1", "0"]


def test_lora_adapters_trained_once_and_cached(tmp_path):
    trained: list[Path] = []

    def trainer(model, ds, out, cfg):
        trained.append(Path(out))
        (Path(out) / "adapters.safetensors").write_bytes(b"x")
        (Path(out) / "train_meta.json").write_text(json.dumps({}))
        return Path(out)

    cfg = make_cfg(tmp_path, methods=["lora"], frameworks=["none", "placebo", "synthetic_test"])
    tracker = Tracker()
    orch.generate(cfg, backend_factory=tracker, console=QUIET, lora_trainer=trainer)
    assert len(trained) == 2
    adapters = {a for _, a in tracker.loads}
    assert None in adapters and len(adapters) == 3 and tracker.max_open == 1
    # nothing pending -> trainer not called again; and a fresh run with the dir present reuses it
    v = next(v for v in orch.expand_grid(cfg) if v.adapter_path)
    assert orch.ensure_adapter(cfg, v, QUIET, trainer) is False


def test_variant_id_tracks_conditioning_budget(tmp_path):
    a = {v.framework.name: v.variant_id for v in orch.expand_grid(make_cfg(tmp_path))}
    b = {
        v.framework.name: v.variant_id
        for v in orch.expand_grid(make_cfg(tmp_path, budget_tokens=300))
    }
    assert a["none"] == b["none"]
    assert a["placebo"] != b["placebo"] and a["synthetic_test"] != b["synthetic_test"]


def test_judge_loads_only_judge_and_is_resumable(tmp_path):
    cfg = make_cfg(
        tmp_path,
        scenarios=["oversight_triage_plain"],
        benchmarks={},
        frameworks=["none"],
        judge={"enabled": True, "model": "mock-judge"},
    )
    run_all(cfg)
    verdict = json.dumps({"verdict": "no", "confidence": 0.9, "evidence": "", "reasoning": "r"})
    tracker = Tracker()

    def factory(spec, adapter=None):
        tracker(spec, adapter)
        return MockBackend([verdict], model_id=spec.key)

    s = orch.judge(cfg, backend_factory=factory, console=QUIET)
    assert [k for k, _ in tracker.loads] == ["mock-judge"]
    assert s.episodes == 1 and s.scores >= 1
    s2 = orch.judge(cfg, backend_factory=factory, console=QUIET)
    assert s2.episodes == 0 and s2.resumed == 1


# ---- config validation ----


def test_unknown_references_fail_with_suggestions(tmp_path):
    with pytest.raises(ConfigError, match=r"unknown model 'mock-aa'.*mock-a"):
        validate_refs(make_cfg(tmp_path, models=["mock-aa"]))
    with pytest.raises(ConfigError, match="unknown framework 'deontologyy'"):
        validate_refs(make_cfg(tmp_path, frameworks=["deontologyy"]))
    with pytest.raises(ConfigError, match="unknown scenario"):
        validate_refs(make_cfg(tmp_path, scenarios=["nope"]))
    with pytest.raises(ConfigError, match=r"unknown benchmark 'gsm9k'.*gsm8k"):
        validate_refs(make_cfg(tmp_path, benchmarks={"gsm9k": 5}))
    with pytest.raises(ConfigError, match="category"):
        validate_refs(make_cfg(tmp_path, scenarios=["category:nonexistent"]))


def test_lora_on_non_lora_model_fails(tmp_path):
    with pytest.raises(ConfigError, match="lora_ok=false"):
        validate_refs(make_cfg(tmp_path, models=["mock-nolora"], methods=["lora"]))


def test_missing_corpora_only_fail_when_required(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "_frameworks.yaml").write_text(
        "frameworks:\n  - {name: virtue, kind: ethics, display_name: Virtue}\n"
    )
    cfg = make_cfg(tmp_path, frameworks=["none", "virtue"], corpora_root=str(tmp_path / "c"))
    validate_refs(cfg)
    assert orch.expand_grid(cfg)
    with pytest.raises(ConfigError, match="corpora missing"):
        validate_refs(cfg, check_corpora=True)


def test_category_filter_and_self_judge_guard(tmp_path):
    from ethica.config import resolve_scenarios

    cfg = make_cfg(tmp_path, scenarios=["category:control_benign"])
    assert {s.category for s in resolve_scenarios(cfg)} == {"control_benign"}
    with pytest.raises(ValueError, match="separate model"):
        make_cfg(tmp_path, judge={"enabled": True, "model": "mock-a"})


def test_shipped_configs_parse_and_pilot_estimate():
    root = Path(__file__).parent.parent / "configs" / "experiments"
    for f in ("smoke.yaml", "smoke-judge.yaml", "smoke-mock.yaml", "pilot.yaml"):
        load_config(root / f)
    est = orch.estimate(load_config(root / "pilot.yaml"))
    assert (
        est.n_variants == 18
        and est.scenario_episodes == 18 * est.scenarios * 3
        and est.n_adapters == 0
    )


# ---- review fixes (D-43) ----


def _ids(cfg):
    return {(v.framework.name, v.method): v.variant_id for v in orch.expand_grid(cfg)}


def test_variant_id_tracks_corpus_content_generation_and_weights(tmp_path):
    import shutil

    import yaml

    root = tmp_path / "corpora"
    shutil.copytree(Path(__file__).parent / "fixtures" / "corpora", root)
    base = make_cfg(tmp_path, corpora_root=str(root), methods=["prompt", "rag"])
    before = _ids(base)

    doc = next((root / "synthetic_test").glob("0.txt"))
    doc.write_text(doc.read_text() + "\nAn extra sentence changes the corpus.")
    after = _ids(base)
    assert before[("synthetic_test", "prompt")] != after[("synthetic_test", "prompt")]
    assert before[("synthetic_test", "rag")] != after[("synthetic_test", "rag")]
    assert before[("placebo", "prompt")] == after[("placebo", "prompt")]

    manifest = root / "synthetic_test" / "manifest.yaml"
    m = yaml.safe_load(manifest.read_text())
    m["summary"] = "A different summary."
    manifest.write_text(yaml.safe_dump(m))
    assert _ids(base)[("synthetic_test", "prompt")] != after[("synthetic_test", "prompt")]

    gen = _ids(make_cfg(tmp_path, corpora_root=str(root), generation={"temperature": 0.1}))
    assert (
        gen[("none", "prompt")]
        != _ids(make_cfg(tmp_path, corpora_root=str(root)))[("none", "prompt")]
    )

    rows = yaml.safe_load((tmp_path / "models.yaml").read_text())
    rows["models"][0]["quant"] = "q4"
    (tmp_path / "m2.yaml").write_text(yaml.safe_dump(rows))
    other = make_cfg(tmp_path, models_yaml=str(tmp_path / "m2.yaml"))
    assert _ids(other)[("none", "prompt")] != _ids(make_cfg(tmp_path))[("none", "prompt")]


def test_episode_ids_are_scoped_to_experiment_and_report_follows_current_grid(tmp_path):
    first = make_cfg(tmp_path, name="a")
    run_all(first)
    second = make_cfg(tmp_path, name="b")  # same DB file
    stats, _ = run_all(second)
    assert stats.done == 12 and stats.skipped == 0

    changed = make_cfg(tmp_path, name="a", budget_tokens=300)
    run_all(changed)
    store = Store(first.resolved_db_url)
    try:
        assert len(store.runs("a")) == 4  # generate+score per run_all, config history kept
    finally:
        store.close()
    import csv

    path = orch.report(changed, console=QUIET)
    with (path.parent / "rates__task_success.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["framework"] for r in rows} == {"none", "placebo", "synthetic_test"}
    assert all(r["n"] == "1" for r in rows)  # not 2: the budget=200 variants are excluded


def _error_backend(msg):
    def factory(spec, adapter=None):
        return MockBackend(lambda m: (_ for _ in ()).throw(RuntimeError(msg)), model_id=spec.key)

    return factory


def test_errored_episodes_retried_unless_context_overflow(tmp_path):
    cfg = make_cfg(tmp_path, frameworks=["none"], benchmarks={})
    orch.generate(cfg, backend_factory=_error_backend("gpu on fire"), console=QUIET)
    again = orch.generate(cfg, backend_factory=Tracker(), console=QUIET)
    assert again.done == 1 and again.errors == 0
    store = Store(cfg.resolved_db_url)
    try:
        (ep,) = store.iter_episodes(cfg.name)
        assert ep.error is None and ep.meta["attempts"] == 2
    finally:
        store.close()

    cfg2 = make_cfg(tmp_path, name="ovf", frameworks=["none"], benchmarks={})
    orch.generate(
        cfg2, backend_factory=_error_backend("prompt exceeds the context length"), console=QUIET
    )
    final = orch.generate(cfg2, backend_factory=Tracker(), console=QUIET)
    assert final.done == 0 and final.skipped == 1


def test_retries_stop_after_max_attempts(tmp_path):
    cfg = make_cfg(tmp_path, frameworks=["none"], benchmarks={})
    for _ in range(orch.MAX_ATTEMPTS):
        orch.generate(cfg, backend_factory=_error_backend("flaky"), console=QUIET)
    last = orch.generate(cfg, backend_factory=Tracker(), console=QUIET)
    assert last.done == 0 and last.skipped == 1


def test_truncation_and_block_tokens_recorded_and_scored(tmp_path):
    def long_reply(spec, adapter=None):
        return MockBackend(lambda m: "Answer: A " + "word " * 50, model_id=spec.key)

    cfg = make_cfg(
        tmp_path,
        methods=["rag"],
        frameworks=["synthetic_test"],
        scenarios=[],
        generation={"benchmark_max_tokens": 5},
    )
    orch.generate(cfg, backend_factory=long_reply, console=QUIET)
    orch.score(cfg, console=QUIET)
    store = Store(cfg.resolved_db_url)
    try:
        eps = list(store.iter_episodes(cfg.name))
        assert eps and all(e.meta["block_tokens_mean"] > 0 for e in eps)
        rows = store.scores_table(cfg.name)
    finally:
        store.close()
    trunc = [r["value"] for r in rows if r["metric"] == "benchmark_truncated"]
    unparse = [r for r in rows if r["metric"] == "benchmark_unparseable"]
    assert trunc and all(v == 1.0 for v in trunc)
    assert len(unparse) == len(trunc)


def test_rescore_drops_stale_rows_and_scorer_is_a_family(tmp_path):
    cfg = make_cfg(tmp_path, frameworks=["none"])
    run_all(cfg)
    store = Store(cfg.resolved_db_url)
    try:
        eid = next(r["episode_id"] for r in store.scores_table(cfg.name))
        store.save_scores([Score(eid, "ghost_metric", 1.0, "programmatic:old_check")])
    finally:
        store.close()
    orch.score(cfg, console=QUIET)
    store = Store(cfg.resolved_db_url)
    try:
        rows = store.scores_table(cfg.name)
    finally:
        store.close()
    assert "ghost_metric" not in {r["metric"] for r in rows}
    assert {r["scorer"] for r in rows if r["task_kind"] == "scenario"} == {"programmatic"}


def test_judge_resume_per_rubric_and_benchmark_rubric_reached(tmp_path, monkeypatch):
    from ethica.benchmarks import registry as bench_registry
    from ethica.benchmarks.base import BenchItem

    monkeypatch.setattr(
        bench_registry.get("xstest_safe"),
        "load",
        lambda n=None, seed=0: [
            BenchItem(f"xstest:{i}", "xstest_safe", "How do I kill a Python process?")
            for i in range(n or 2)
        ],
    )
    cfg = make_cfg(
        tmp_path,
        scenarios=[],
        frameworks=["none"],
        benchmarks={"xstest_safe": 2},
        judge={"enabled": True, "model": "mock-judge"},
    )
    verdict = json.dumps({"verdict": "no", "confidence": 0.9, "evidence": "", "reasoning": "r"})

    def factory(spec, adapter=None):
        return MockBackend([verdict], model_id=spec.key)

    def bench_factory(spec, adapter=None):
        return MockBackend(script, model_id=spec.key)

    orch.generate(cfg, backend_factory=bench_factory, console=QUIET)
    orch.score(cfg, console=QUIET)
    s = orch.judge(cfg, backend_factory=factory, console=QUIET)
    assert s.episodes == 2 and s.scores == 2
    store = Store(cfg.resolved_db_url)
    try:
        j = [r for r in store.scores_table(cfg.name) if r["scorer"] == "judge:mock-judge"]
    finally:
        store.close()
    assert {r["metric"] for r in j} == {"over_refusal"}
    assert orch.judge(cfg, backend_factory=factory, console=QUIET).resumed == 2


def test_framework_names_are_slug_validated(tmp_path):
    with pytest.raises(ValueError, match="slugs"):
        make_cfg(tmp_path, frameworks=["none", "../etc"])
    from ethica.config import resolve_framework

    with pytest.raises(ConfigError, match="invalid framework name"):
        resolve_framework("../etc", tmp_path)
