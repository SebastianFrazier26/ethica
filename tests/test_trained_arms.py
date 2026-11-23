import csv
import json

import pytest
from orch_helpers import Tracker, make_cfg, script
from rich.console import Console

from ethica import orchestrator as orch
from ethica.conditioning.factory import make_conditioner
from ethica.conditioning.prompt import NoneConditioner
from ethica.config import ConfigError, validate_refs

QUIET = Console(quiet=True)


def _adapter(tmp_path, name, wall=1.0):
    d = tmp_path / "adapters" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapters.safetensors").write_bytes(b"x")
    (d / "train_meta.json").write_text(json.dumps({"wall_s": wall}))
    return str(d)


def _cfg(tmp_path, **over):
    arms = [
        {
            "name": "generic",
            "kind": "placebo",
            "model": "mock-a",
            "adapter": _adapter(tmp_path, "g"),
        },
        {"name": "virtue", "kind": "ethics", "model": "mock-a", "adapter": _adapter(tmp_path, "v")},
    ]
    base = {
        "frameworks": ["none"],
        "methods": ["none"],
        "trained_arms": arms,
        "report": {"baseline": "generic", "min_tasks": 1, "n_boot": 50, "n_perm": 50},
    }
    return make_cfg(tmp_path, **{**base, **over})


def test_grid_adds_trained_arms_with_no_conditioning(tmp_path):
    cfg = _cfg(tmp_path)
    grid = orch.expand_grid(cfg)
    assert [(v.framework.name, v.framework.kind, v.method) for v in grid] == [
        ("none", "none", "none"),
        ("generic", "placebo", "none"),
        ("virtue", "ethics", "none"),
    ]
    assert grid[0].adapter_path is None
    assert grid[2].adapter_path == cfg.trained_arms[1].adapter
    assert len({v.variant_id for v in grid}) == 3
    for v in grid:
        assert isinstance(
            make_conditioner(v, cfg.corpora_root, cfg.budget_tokens, cfg.rag_k), NoneConditioner
        )


def test_retrained_adapter_is_a_new_variant(tmp_path):
    cfg = _cfg(tmp_path)
    before = {v.framework.name: v.variant_id for v in orch.expand_grid(cfg)}
    _adapter(tmp_path, "v", wall=2.0)
    after = {v.framework.name: v.variant_id for v in orch.expand_grid(cfg)}
    assert after["virtue"] != before["virtue"]
    assert after["generic"] == before["generic"] and after["none"] == before["none"]


def test_no_trained_arms_leaves_the_grid_unchanged(tmp_path):
    plain = make_cfg(tmp_path)
    assert plain.trained_arms == []
    with_arms = make_cfg(
        tmp_path,
        methods=["prompt", "none"],
        trained_arms=[
            {
                "name": "virtue",
                "kind": "ethics",
                "model": "mock-a",
                "adapter": _adapter(tmp_path, "v"),
            }
        ],
    )
    ids = [v.variant_id for v in orch.expand_grid(plain)]
    assert [v.variant_id for v in orch.expand_grid(with_arms)][: len(ids)] != []
    assert set(ids) <= {v.variant_id for v in orch.expand_grid(with_arms)}


@pytest.mark.parametrize("missing", ["adapters.safetensors", "train_meta.json"])
def test_missing_adapter_fails_before_any_model_loads(tmp_path, missing):
    cfg = _cfg(tmp_path)
    (tmp_path / "adapters" / "v" / missing).unlink()
    tracker = Tracker()
    with pytest.raises(ConfigError, match=rf"trained arm 'virtue'.*{missing}"):
        orch.generate(cfg, backend_factory=tracker, console=QUIET)
    assert tracker.loads == []


def test_config_errors(tmp_path):
    arm = {
        "name": "virtue",
        "kind": "ethics",
        "model": "mock-b",
        "adapter": _adapter(tmp_path, "v"),
    }
    with pytest.raises(ConfigError, match="model 'mock-b' is not in `models`"):
        validate_refs(make_cfg(tmp_path, frameworks=["none"], methods=["none"], trained_arms=[arm]))
    arm["model"] = "mock-a"
    with pytest.raises(ConfigError, match="add it to `methods`"):
        validate_refs(make_cfg(tmp_path, frameworks=["none"], trained_arms=[arm]))
    arm["name"] = "placebo"
    with pytest.raises(ConfigError, match="unique and differ from frameworks"):
        validate_refs(
            make_cfg(tmp_path, frameworks=["none", "placebo"], methods=["none"], trained_arms=[arm])
        )


def test_end_to_end_loads_adapters_and_reports_against_generic(tmp_path):
    cfg = _cfg(tmp_path)
    tracker = Tracker()

    def factory(spec, adapter=None):
        b = tracker(spec, adapter)
        b._script = script
        return b

    stats = orch.generate(cfg, backend_factory=factory, console=QUIET)
    assert stats.errors == 0 and stats.adapters_trained == 0
    assert sorted(tracker.loads, key=str) == sorted(
        [("mock-a", None), *(("mock-a", a.adapter) for a in cfg.trained_arms)], key=str
    )
    assert tracker.max_open == 1
    orch.score(cfg, console=QUIET)
    report = orch.report(cfg, console=QUIET)
    with next(report.parent.glob("contrasts__fidelity__*.csv")).open() as fh:
        rows = list(csv.DictReader(fh))
    assert {(r["framework"], r["reference"]) for r in rows} == {
        ("virtue", "generic"),
        ("virtue", "none"),
    }


def test_weights_key_tracks_recorded_tokenizer_edits_only(tmp_path):
    from ethica.types import ModelSpec

    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    spec = ModelSpec("m", str(d), "f", 1.0, "base")
    cfg = make_cfg(tmp_path)
    plain = orch.weights_key(cfg, spec)
    (d / "provenance.json").write_text(json.dumps({"source_repo": "o/m"}))
    assert orch.weights_key(cfg, spec) == plain  # v0-style provenance: id unchanged
    (d / "provenance.json").write_text(json.dumps({"tokenizer_edits": {"chat_format": "plain"}}))
    a = orch.weights_key(cfg, spec)
    (d / "provenance.json").write_text(json.dumps({"tokenizer_edits": {"chat_format": "chatml"}}))
    assert len({plain, a, orch.weights_key(cfg, spec)}) == 3


def test_pressure_variants_read_the_contrary_corpus(tmp_path):
    from ethica.conditioning.prompt import PromptConditioner
    from ethica.types import Message

    arms = [
        {
            "name": "virtue",
            "kind": "ethics",
            "model": "mock-a",
            "adapter": _adapter(tmp_path, "v"),
            "pressure": ["synthetic_test"],
        }
    ]
    cfg = make_cfg(tmp_path, frameworks=["none"], methods=["none"], trained_arms=arms)
    validate_refs(cfg, check_corpora=True)
    grid = orch.expand_grid(cfg)
    pressed = grid[-1]
    assert (pressed.framework.name, pressed.framework.kind, pressed.method) == (
        "virtue+under+synthetic_test",
        "ethics",
        "prompt",
    )
    assert pressed.adapter_path == arms[0]["adapter"]
    assert pressed.variant_id != grid[-2].variant_id
    cond = make_conditioner(pressed, cfg.corpora_root, cfg.budget_tokens, cfg.rag_k)
    assert isinstance(cond, PromptConditioner)
    system = cond.prepare([Message("user", "hi")])[0].content
    assert "<background_reading>" in system

    arms[0]["pressure"] = ["none"]
    with pytest.raises(ConfigError, match="pressure needs a framework corpus"):
        validate_refs(make_cfg(tmp_path, frameworks=["none"], methods=["none"], trained_arms=arms))
    arms[0]["pressure"] = ["no_such_framework"]
    with pytest.raises(ConfigError, match="unknown framework"):
        validate_refs(make_cfg(tmp_path, frameworks=["none"], methods=["none"], trained_arms=arms))
