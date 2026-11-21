import math

import pytest
from sqlalchemy.exc import IntegrityError

from ethica.tracking.store import Store
from ethica.types import (
    EnvEvent,
    EpisodeResult,
    FrameworkSpec,
    Message,
    ModelSpec,
    Score,
    ToolCall,
    Variant,
)


def make_variant(vid="v1", safeguards="stock", fw="placebo", kind="placebo", method="prompt"):
    return Variant(
        variant_id=vid,
        model=ModelSpec("m1", "org/m1", "qwen", 4.0, safeguards, "mock"),
        framework=FrameworkSpec(fw, kind, fw.title(), None),
        method=method,
    )


def make_episode(eid="e1", vid="v1", task="t1", seed=0, error=None):
    return EpisodeResult(
        episode_id=eid,
        variant_id=vid,
        task_id=task,
        task_kind="scenario",
        seed=seed,
        transcript=[
            Message("system", "sys"),
            Message("assistant", "hi", tool_calls=[ToolCall("ls", {"p": "/"}, "c1")]),
            Message("tool", "out", tool_call_id="c1", name="ls"),
        ],
        events=[EnvEvent(0, "ls", {"p": "/"}, "out", True, frozenset({"a", "b"}))],
        final_env_state={"files": ["x"]},
        n_turns=2,
        prompt_tokens=10,
        completion_tokens=5,
        wall_s=1.5,
        error=error,
        meta={"k": 1},
    )


@pytest.fixture
def store():
    s = Store("sqlite://")
    s.upsert_variant(make_variant())
    return s


def test_default_url_creates_parent_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHICA_DB_URL", f"sqlite:///{tmp_path}/sub/dir/x.db")
    s = Store()
    assert (tmp_path / "sub/dir/x.db").exists()
    s.close()


def test_roundtrip(store):
    eid = store.create_experiment("exp", {"a": 1}, git_rev="abc")
    ep = make_episode()
    assert store.save_episode(ep, eid) is True
    [back] = list(store.iter_episodes(experiment_id=eid))
    assert back == ep


def test_idempotent_and_resume(store):
    eid = store.create_experiment("exp")
    other = store.create_experiment("other")
    assert store.save_episode(make_episode(), eid)
    assert not store.save_episode(make_episode(), eid)
    store.save_episode(make_episode("e2", task="t2"), eid)
    store.save_episode(make_episode("e3", task="t3"), other)
    assert store.has_episode("e1") and not store.has_episode("zz")
    assert store.completed_episode_ids(eid) == {"e1", "e2"}
    assert len(list(store.iter_episodes(experiment_id=eid, task_id="t2"))) == 1
    assert store.episode_counts(eid) == {"episodes": 2, "errors": 0}


def test_fk_enforced(store):
    eid = store.create_experiment("exp")
    with pytest.raises(IntegrityError):
        store.save_episode(make_episode(vid="missing"), eid)
    assert not store.has_episode("e1")


def test_scores_upsert_and_table(store):
    eid = store.create_experiment("exp")
    store.upsert_variant(make_variant("v2", "abliterated", "deontology", "ethics"))
    store.save_episode(make_episode("e1", "v1"), eid)
    store.save_episode(make_episode("e2", "v2", error="boom"), eid)
    store.save_scores(
        [
            Score("e1", "misaligned_action", 1.0, "programmatic:x"),
            Score("e1", "misaligned_action", 0.0, "judge:j"),
            Score("e2", "misaligned_action", math.nan, "programmatic:x"),
            Score("e2", "task_success", 1.0, "programmatic:y"),
        ]
    )
    store.save_scores([Score("e1", "misaligned_action", 0.0, "programmatic:x", "rescored")])
    rows = store.scores_table(eid, ["misaligned_action"])
    assert len(rows) == 3
    prog = next(r for r in rows if r["episode_id"] == "e1" and r["scorer"] == "programmatic:x")
    assert prog["value"] == 0.0
    assert prog["model_key"] == "m1" and prog["framework"] == "placebo"
    assert prog["safeguards"] == "stock" and prog["task_id"] == "t1" and prog["seed"] == 0
    nan_row = next(r for r in rows if r["episode_id"] == "e2")
    assert math.isnan(nan_row["value"]) and nan_row["errored"]
    assert len(store.scores_table(eid)) == 4
    assert all(not r["errored"] for r in store.scores_table(eid, include_errors=False))
    assert store.scores_table("nope") == []


def test_upsert_variant_does_not_overwrite_provenance(store):
    first = make_variant()
    changed = Variant(
        first.variant_id,
        ModelSpec("m1", "org/OTHER", "qwen", 4.0, "stock", "mock"),
        first.framework,
        first.method,
        adapter_path="/new",
    )
    store.upsert_variant(changed)
    eid = store.create_experiment("t")
    store.save_episode(make_episode(), eid)
    store.save_scores([Score("e1", "m", 1.0, "programmatic")])
    (row,) = store.scores_table(eid)
    assert row["variant_id"] == "v1"
    with store._session() as s:
        from ethica.tracking.models import Variant as VariantRow

        v = s.get(VariantRow, "v1")
        assert v.hf_repo == "org/m1" and v.adapter_path is None


def test_pair_key_is_base_of_or_own_key():
    s = Store("sqlite://")
    abl = Variant(
        "va",
        ModelSpec("m1-abl", "o/m1-abl", "qwen", 4.0, "abliterated", "mock", base_of="m1"),
        FrameworkSpec("none", "none", "None", None),
        "prompt",
    )
    s.upsert_variant(abl)
    s.upsert_variant(make_variant())
    eid = s.create_experiment("t")
    for v in ("va", "v1"):
        s.save_episode(make_episode(f"e-{v}", v), eid)
        s.save_scores([Score(f"e-{v}", "m", 1.0, "programmatic")])
    assert {r["model_key"]: r["pair_key"] for r in s.scores_table(eid)} == {
        "m1-abl": "m1",
        "m1": "m1",
    }


def test_rescore_replaces_same_family_only(store):
    eid = store.create_experiment("t")
    store.save_episode(make_episode(), eid)
    store.save_scores(
        [
            Score("e1", "old_metric", 1.0, "programmatic"),
            Score("e1", "kept", 1.0, "programmatic"),
            Score("e1", "j", 1.0, "judge:x"),
            Score("e1", "h", 1.0, "heuristic:refusal"),
        ]
    )
    store.save_scores([Score("e1", "kept", 0.0, "programmatic:legacy-name")], replace_families=True)
    got = {(r["metric"], r["scorer"]) for r in store.scores_table(eid)}
    assert got == {
        ("kept", "programmatic:legacy-name"),
        ("j", "judge:x"),
        ("h", "heuristic:refusal"),
    }


def test_episode_status_delete_and_run_history(store):
    eid = store.create_experiment("t")
    ep = make_episode(error="boom")
    ep.meta["attempts"] = 2
    store.save_episode(ep, eid)
    assert store.episode_status(eid) == {"e1": ("boom", 2)}
    store.delete_episode("e1")
    assert store.episode_status(eid) == {} and store.episode_counts(eid)["episodes"] == 0
    store.record_run(eid, "generate", {"a": 1})
    store.record_run(eid, "score", {"a": 2})
    assert [(r.phase, r.config_json) for r in store.runs(eid)] == [
        ("generate", '{"a": 1}'),
        ("score", '{"a": 2}'),
    ]


def test_scorer_family():
    from ethica.tracking.store import scorer_family

    assert scorer_family("programmatic:tag_present,tool_called") == "programmatic"
    assert scorer_family("exact") == "exact"
    assert scorer_family("judge:qwen") == "judge:qwen"
    assert scorer_family("heuristic:refusal") == "heuristic:refusal"
