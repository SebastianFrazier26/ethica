"""Model registry: configs/models.yaml -> validated ModelSpecs, plus backend factory."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ethica.types import ChatBackend, ModelSpec, Safeguards

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"
OPENAI_BASE_URL_ENV = "ETHICA_OPENAI_BASE_URL"
# Weights, configs, tokenizer and chat-template files only. mlx_lm's own default also allows
# *.py (remote-code files), which we never want fetched; no .bin/.pt (pickle) either.
MODEL_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


class ModelEntry(BaseModel):
    """YAML row: the ModelSpec fields plus provenance that the experiment code does not need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]*$")
    hf_repo: str = Field(pattern=r"^[\w.\-]+/[\w.\-]+$")
    family: str
    params_b: float = Field(gt=0)
    safeguards: Safeguards
    backend: Literal["mlx", "openai", "mock"] = "mlx"
    base_of: str | None = None
    max_context: int = Field(default=8192, gt=0)
    lora_ok: bool = True
    notes: str = ""
    size_gb: float = Field(gt=0)
    # repo needs a licence click-through for the *listed* hf_repo (mirrors are usually ungated)
    gated: bool = False
    # False: weights are HF-format; mlx_lm.load still works but `mlx_lm.convert -q` is advised
    mlx_native: bool = True
    quant: str = "bf16"
    # Locally converted weights (scripts/harmonize_quant.py), relative to the repo root. Preferred
    # over hf_repo at load time; hf_repo stays as the provenance record. Existence is checked in
    # to_spec, not at parse time, because models/ is gitignored and absent on a fresh clone/CI.
    local_path: str | None = None
    # Hub commit sha of hf_repo (security M2: a moved branch must not change the weights). Ignored
    # when local_path is set.
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def _check_lineage(self) -> Self:
        if self.safeguards == "abliterated" and not self.base_of:
            raise ValueError(f"{self.key}: abliterated model must set base_of")
        if self.safeguards == "stock" and self.base_of:
            raise ValueError(f"{self.key}: stock model must not set base_of")
        if self.safeguards == "base" and self.base_of:
            raise ValueError(f"{self.key}: base model must not set base_of")
        return self

    def resolved_local_path(self) -> Path | None:
        if self.local_path is None:
            return None
        return (REPO_ROOT / self.local_path).resolve()

    def to_spec(self, check_local: bool = True) -> ModelSpec:
        # ModelSpec has no local-path field; mlx_lm.load / lora accept a directory in hf_repo
        local = self.resolved_local_path()
        if local is not None and check_local and not (local / "config.json").is_file():
            raise FileNotFoundError(
                f"{self.key}: local_path {self.local_path!r} missing; "
                f"run scripts/harmonize_quant.py {self.key}"
            )
        return ModelSpec(
            key=self.key,
            hf_repo=str(local) if local is not None else self.hf_repo,
            family=self.family,
            params_b=self.params_b,
            safeguards=self.safeguards,
            backend=self.backend,
            base_of=self.base_of,
            max_context=self.max_context,
            lora_ok=self.lora_ok,
            notes=self.notes,
            revision=None if local is not None else self.revision,
        )


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _check_unique_and_links(self) -> Self:
        keys = [m.key for m in self.models]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise ValueError(f"duplicate model keys: {sorted(dupes)}")
        by_key = {m.key: m for m in self.models}
        for m in self.models:
            if m.base_of is not None:
                base = by_key.get(m.base_of)
                if base is None or base.safeguards != "stock":
                    raise ValueError(f"{m.key}: base_of={m.base_of!r} is not a stock model in file")
        return self


def load_entries(path: Path | str | None = None) -> dict[str, ModelEntry]:
    p = Path(path or os.environ.get("ETHICA_MODELS_YAML") or DEFAULT_MODELS_YAML)
    return _load_cached(str(p))


@lru_cache(maxsize=8)
def _load_cached(path: str) -> dict[str, ModelEntry]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {m.key: m for m in _File.model_validate(raw).models}


def get_entry(key: str, path: Path | str | None = None) -> ModelEntry:
    entries = load_entries(path)
    if key not in entries:
        raise KeyError(f"unknown model {key!r}; known: {sorted(entries)}")
    return entries[key]


def get_model(key: str, path: Path | str | None = None) -> ModelSpec:
    return get_entry(key, path).to_spec()


def list_models(
    family: str | None = None,
    safeguards: Safeguards | None = None,
    path: Path | str | None = None,
) -> list[ModelSpec]:
    return [
        e.to_spec(check_local=False)
        for e in load_entries(path).values()
        if (family is None or e.family == family)
        and (safeguards is None or e.safeguards == safeguards)
    ]


def make_backend(spec: ModelSpec, adapter_path: str | None = None) -> ChatBackend:
    # imports are local so that importing the registry never pulls in mlx
    if spec.backend == "mlx":
        from ethica.backends.mlx_backend import MLXBackend

        return MLXBackend(spec, adapter_path=adapter_path)
    if adapter_path is not None:
        raise ValueError(f"adapter_path is only supported by the mlx backend, not {spec.backend}")
    if spec.backend == "openai":
        from ethica.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(
            base_url=os.environ.get(OPENAI_BASE_URL_ENV, "http://localhost:8080"),
            model=spec.hf_repo,
        )
    from ethica.backends.mock import MockBackend

    return MockBackend(["ok"], model_id=spec.key)


def pull(key: str, path: Path | str | None = None) -> str:
    """Download the model's HF repo into the HF cache; returns the local snapshot dir."""
    spec = get_model(key, path)
    return snapshot(spec, token=os.environ.get("HF_TOKEN") or None)


def snapshot(spec: ModelSpec, token: str | None = None) -> str:
    """Resolve a spec to a local directory: pinned revision, restricted file patterns."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec.hf_repo,
        revision=spec.revision,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        token=token,
    )
