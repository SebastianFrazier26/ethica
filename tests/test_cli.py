import json
import shutil
from pathlib import Path

import yaml
from orch_helpers import write_models_yaml
from typer.testing import CliRunner

from ethica import cli
from ethica.cli import app

runner = CliRunner()
ROOT = Path(__file__).parent.parent
CORPORA = str(Path(__file__).parent / "fixtures" / "corpora")


def mock_config(tmp_path: Path, **over: object) -> Path:
    cfg = {
        "name": "cli-t",
        "models": ["mock-a"],
        "models_yaml": write_models_yaml(tmp_path / "models.yaml"),
        "frameworks": ["none", "synthetic_test"],
        "methods": ["prompt"],
        "scenarios": ["control_inbox_report"],
        "benchmarks": {"fidelity": 2},
        "budget_tokens": 200,
        "corpora_root": CORPORA,
        "out_dir": str(tmp_path / "out"),
        "report": {"min_tasks": 1, "n_boot": 50, "n_perm": 50},
        **over,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_all_phases_with_mock_backend(tmp_path):
    cfg = mock_config(tmp_path)
    r = runner.invoke(app, ["run", str(cfg)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "out" / "report" / "report.md").is_file()
    assert (tmp_path / "out" / "ethica.db").is_file()


def test_single_phase_and_resume(tmp_path):
    cfg = mock_config(tmp_path)
    assert runner.invoke(app, ["run", str(cfg), "--phase", "generate"]).exit_code == 0
    assert not (tmp_path / "out" / "report").exists()
    assert runner.invoke(app, ["score", str(cfg)]).exit_code == 0
    r = runner.invoke(app, ["run", str(cfg), "--phase", "generate"])
    assert "0 episodes to run" in r.output
    assert runner.invoke(app, ["report", str(cfg)]).exit_code == 0


def test_dry_run_prints_grid_without_corpora():
    r = runner.invoke(app, ["run", str(ROOT / "configs/experiments/pilot.yaml"), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "18 variants" in r.output and "estimated wall time" in r.output


def test_pilot_refuses_real_run_without_corpora(tmp_path):
    # Real corpora may exist locally; point at a root with only the registry so the
    # refusal path is tested without ever dispatching a model-loading run.
    root = tmp_path / "corpora"
    root.mkdir()
    shutil.copy(ROOT / "corpora/_frameworks.yaml", root / "_frameworks.yaml")
    cfg = yaml.safe_load((ROOT / "configs/experiments/pilot.yaml").read_text())
    cfg["corpora_root"] = str(root)
    cfg["out_dir"] = str(tmp_path / "out")
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    r = runner.invoke(app, ["run", str(path)])
    assert r.exit_code == 2 and "corpora missing" in r.output


def test_bad_config_exits_2_with_message(tmp_path):
    cfg = mock_config(tmp_path, models=["nope"])
    r = runner.invoke(app, ["run", str(cfg)])
    assert r.exit_code == 2 and "unknown model" in r.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nbogus: 1\n")
    assert runner.invoke(app, ["run", str(bad)]).exit_code == 2


def test_bad_args_rejected_at_boundary(tmp_path):
    assert runner.invoke(app, ["run", str(tmp_path / "missing.yaml")]).exit_code != 0
    assert runner.invoke(app, ["run", "x.yaml", "--phase", "bogus"]).exit_code != 0
    assert runner.invoke(app, ["models", "list", "--safeguards", "evil"]).exit_code != 0
    assert runner.invoke(app, ["corpus", "check", "../etc"]).exit_code != 0


def test_listing_commands(monkeypatch):
    monkeypatch.setattr(cli.console, "width", 200)
    assert "oversight_triage_plain" in runner.invoke(app, ["scenarios", "list"]).output
    r = runner.invoke(app, ["scenarios", "show", "control_inbox_report"])
    assert r.exit_code == 0 and "system_prompt" in r.output
    assert runner.invoke(app, ["scenarios", "show", "nope"]).exit_code == 2
    assert "fidelity" in runner.invoke(app, ["benchmarks", "list"]).output
    assert "tiny" in runner.invoke(app, ["models", "list"]).output


def test_corpus_commands(monkeypatch):
    monkeypatch.setattr(cli.console, "width", 200)
    r = runner.invoke(app, ["corpus", "check", "synthetic_test", "--root", CORPORA])
    assert r.exit_code == 0 and "3 docs" in r.output
    assert "synthetic_test" in runner.invoke(app, ["corpus", "list", "--root", CORPORA]).output
    assert runner.invoke(app, ["corpus", "check", "unknown", "--root", CORPORA]).exit_code == 2


def test_label_import_and_agreement(tmp_path):
    sheet = tmp_path / "labels.jsonl"
    key = tmp_path / "labels.key.jsonl"
    items = [("a", "yes", "yes"), ("b", "no", "no"), ("c", "yes", "no")]
    sheet.write_text("".join(json.dumps({"item_id": i, "label": h}) + "\n" for i, h, _ in items))
    key.write_text(
        "".join(json.dumps({"item_id": i, "judge_verdict": j}) + "\n" for i, _, j in items)
    )
    assert "3 labelled" in runner.invoke(app, ["label", "import", str(sheet)]).output
    r = runner.invoke(app, ["label", "agreement", str(sheet), str(key)])
    assert r.exit_code == 0 and "kappa" in r.output
