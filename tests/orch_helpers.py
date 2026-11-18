"""Shared fixtures for orchestrator/CLI tests: mock-only model registry, scripted backends."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ethica.backends.mock import MockBackend
from ethica.config import ExperimentConfig
from ethica.types import ChatBackend, Message, ModelSpec

CORPORA = str(Path(__file__).parent / "fixtures" / "corpora")

_MODELS = [
    {"key": "mock-a", "hf_repo": "mock/a", "lora_ok": True},
    {"key": "mock-b", "hf_repo": "mock/b", "lora_ok": True},
    {"key": "mock-nolora", "hf_repo": "mock/n", "lora_ok": False},
    {"key": "mock-judge", "hf_repo": "mock/j", "lora_ok": False},
]


def write_models_yaml(path: Path) -> str:
    rows = [
        {
            "family": "mock",
            "params_b": 0.1,
            "safeguards": "stock",
            "backend": "mock",
            "size_gb": 0.01,
            "quant": "none",
            **m,
        }
        for m in _MODELS
    ]
    path.write_text(yaml.safe_dump({"models": rows}), encoding="utf-8")
    return str(path)


def make_cfg(tmp_path: Path, **over: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "name": "t",
        "models": ["mock-a"],
        "frameworks": ["none", "placebo", "synthetic_test"],
        "methods": ["prompt"],
        "scenarios": ["control_inbox_report"],
        "benchmarks": {"fidelity": 3},
        "seeds": [0],
        "budget_tokens": 200,
        "corpora_root": CORPORA,
        "models_yaml": write_models_yaml(tmp_path / "models.yaml"),
        "out_dir": str(tmp_path / "out"),
        "db_url": f"sqlite:///{tmp_path / 'e.db'}",
        "adapters_dir": str(tmp_path / "adapters"),
        "report": {"min_tasks": 1, "n_boot": 50, "n_perm": 50},
    }
    base.update(over)
    return ExperimentConfig.model_validate(base)


FULL_CALL = json.dumps(
    {
        "name": "write_file",
        "arguments": {
            "path": "/reports/inbox_summary.txt",
            "content": "invoice due; team lunch friday; password rotation",
        },
    }
)


def script(messages: list[Message]) -> str:
    system = messages[0].content if messages and messages[0].role == "system" else ""
    if "## Tools" not in system:
        return "Answer: A"
    if not any(m.role == "assistant" for m in messages):
        return f"<tool_call>{FULL_CALL}</tool_call>"
    return "<done/>"


class Tracker:
    """Backend factory that counts live instances (one-model-resident invariant)."""

    def __init__(self) -> None:
        self.open = 0
        self.max_open = 0
        self.loads: list[tuple[str, str | None]] = []

    def __call__(self, spec: ModelSpec, adapter: str | None = None) -> ChatBackend:
        tracker = self
        tracker.open += 1
        tracker.max_open = max(tracker.max_open, tracker.open)
        tracker.loads.append((spec.key, adapter))

        class _B(MockBackend):
            def close(self) -> None:
                if not self.closed:
                    tracker.open -= 1
                super().close()

        return _B(script, model_id=spec.key)
