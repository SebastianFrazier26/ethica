"""Experiment config: YAML -> validated ExperimentConfig, plus reference resolution.

Two stages on purpose. ``load_config`` checks shape only, so a config can be written before its
corpora exist (the pilot is). ``validate_refs`` resolves every model/framework/scenario/benchmark
name and fails fast with the list of valid choices; ``check_corpora`` is only demanded at run time.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ethica.backends.registry import get_entry, load_entries
from ethica.benchmarks import registry as bench_registry
from ethica.conditioning.corpus import (
    corpus_built,
    framework_from_manifest,
    load_manifest,
    none_framework,
)
from ethica.conditioning.lora import LoRAConfig
from ethica.scenarios.registry import load_registry
from ethica.scenarios.schema import Scenario
from ethica.types import ConditioningMethod, FrameworkSpec, GenerationParams, ModelSpec

ALWAYS_KNOWN_FRAMEWORKS = ("none", "placebo")
_SLUG = r"^[a-z0-9][a-z0-9_.\-]*$"
_FRAMEWORK_SLUG = re.compile(r"[a-z0-9][a-z0-9_+\-]*")


class ConfigError(ValueError):
    """A config references something that does not exist, or combines options illegally."""


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    benchmark_max_tokens: int = Field(default=512, gt=0)
    stop: list[str] = Field(default_factory=list)

    def scenario_params(self) -> GenerationParams:
        return GenerationParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=tuple(self.stop),
        )

    def benchmark_params(self) -> GenerationParams:
        return GenerationParams(
            max_tokens=self.benchmark_max_tokens, temperature=self.temperature, top_p=self.top_p
        )


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "judge-default"
    enabled: bool = False
    rubrics: list[str] | None = None
    # self-judging invalidates scores; only the plumbing smoke test should set this
    allow_self_judge: bool = False


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: str = "placebo"
    min_tasks: int = Field(default=10, ge=1)
    n_boot: int = Field(default=2000, ge=1)
    n_perm: int = Field(default=10000, ge=1)
    # task is the primary unit (D-43): seeds within a task are correlated, and the bootstrap
    # already resamples tasks
    permute_by: Literal["pair", "task"] = "task"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_SLUG)
    models: list[str] = Field(min_length=1)
    frameworks: list[str] = Field(min_length=1)
    methods: list[ConditioningMethod] = Field(default=["prompt"], min_length=1)
    # "all", scenario ids, or "category:<name>"
    scenarios: list[str] = Field(default_factory=lambda: ["all"])
    benchmarks: dict[str, int] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    scratchpad: bool = True
    budget_tokens: int = Field(default=1000, gt=0)
    rag_k: int = Field(default=4, gt=0)
    lora: dict[str, Any] = Field(default_factory=dict)
    lora_target_tokens: int | None = Field(default=None, gt=0)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    models_yaml: str | None = None
    db_url: str | None = None
    corpora_root: str = "corpora"
    out_dir: str | None = None
    adapters_dir: str = "adapters"

    @field_validator("scenarios", mode="before")
    @classmethod
    def _scalar_scenarios(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    @field_validator("benchmarks")
    @classmethod
    def _positive_n(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: n for k, n in v.items() if n <= 0}
        if bad:
            raise ValueError(f"benchmark item counts must be > 0: {bad}")
        return v

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v

    @field_validator("frameworks")
    @classmethod
    def _framework_slugs(cls, v: list[str]) -> list[str]:
        bad = [n for n in v if not _FRAMEWORK_SLUG.fullmatch(n)]
        if bad:
            raise ValueError(f"framework names must be lowercase slugs [a-z0-9_+-]: {bad}")
        return v

    @field_validator("models", "frameworks", "methods")
    @classmethod
    def _no_dupes(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate entries in {v}")
        return v

    @field_validator("lora")
    @classmethod
    def _lora_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        LoRAConfig(**v)
        return v

    @model_validator(mode="after")
    def _judge_not_under_test(self) -> Self:
        if (
            self.judge.enabled
            and not self.judge.allow_self_judge
            and self.judge.model in self.models
        ):
            raise ValueError(
                f"judge model {self.judge.model!r} is also under test; the judge must be a "
                "separate model"
            )
        return self

    @property
    def experiment_id(self) -> str:
        return self.name

    @property
    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir else Path("runs") / self.name

    @property
    def resolved_db_url(self) -> str:
        return self.db_url or f"sqlite:///{self.resolved_out_dir / 'ethica.db'}"

    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(**self.lora)


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{p}: cannot read config: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValueError as e:
        raise ConfigError(f"{p}: {e}") from e


# ---------- reference resolution ----------


def _suggest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=3)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    return f"{hint} Known: {', '.join(sorted(known))}"


class _FrameworkRow(BaseModel):
    name: str
    kind: Literal["none", "placebo", "ethics"]
    display_name: str
    summary: str = ""


class _FrameworkFile(BaseModel):
    frameworks: list[_FrameworkRow]


def _framework_rows(corpora_root: Path) -> dict[str, _FrameworkRow]:
    f = corpora_root / "_frameworks.yaml"
    if not f.is_file():
        return {}
    rows = _FrameworkFile.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
    return {r.name: r for r in rows.frameworks}


def _corpus_dirs(corpora_root: Path) -> list[str]:
    if not corpora_root.is_dir():
        return []
    return sorted(
        d.name
        for d in corpora_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "manifest.yaml").is_file()
    )


def known_frameworks(corpora_root: str | Path) -> list[str]:
    root = Path(corpora_root)
    names = {*ALWAYS_KNOWN_FRAMEWORKS, *_framework_rows(root), *_corpus_dirs(root)}
    return sorted(names)


def resolve_framework(name: str, corpora_root: str | Path) -> FrameworkSpec:
    """Manifest on disk wins; otherwise the declared row in ``_frameworks.yaml``."""
    if name == "none":
        return none_framework()
    if not _FRAMEWORK_SLUG.fullmatch(name):
        raise ConfigError(f"invalid framework name {name!r}: want a lowercase slug [a-z0-9_+-]")
    root = Path(corpora_root)
    cdir = root / name
    if (cdir / "manifest.yaml").is_file():
        return framework_from_manifest(load_manifest(cdir), cdir.resolve())
    rows = _framework_rows(root)
    if name in rows:
        r = rows[name]
        return FrameworkSpec(
            name=name,
            kind=r.kind,
            display_name=r.display_name,
            corpus_dir=str(cdir.resolve()),
            summary=r.summary.strip(),
        )
    raise ConfigError(f"unknown framework {name!r}.{_suggest(name, known_frameworks(root))}")


def resolve_model(key: str, cfg: ExperimentConfig) -> ModelSpec:
    entries = load_entries(cfg.models_yaml)
    if key not in entries:
        raise ConfigError(f"unknown model {key!r}.{_suggest(key, list(entries))}")
    return get_entry(key, cfg.models_yaml).to_spec()


def resolve_scenarios(cfg: ExperimentConfig) -> list[Scenario]:
    reg = load_registry()
    out: dict[str, Scenario] = {}
    for sel in cfg.scenarios:
        if sel == "all":
            picked = reg.list()
        elif sel.startswith("category:"):
            cat = sel.removeprefix("category:")
            picked = reg.list(category=cat)
            if not picked:
                cats = sorted({s.category for s in reg.list()})
                raise ConfigError(f"no scenarios in category {cat!r}. Categories: {cats}")
        else:
            try:
                picked = [reg.get(sel)]
            except KeyError:
                raise ConfigError(
                    f"unknown scenario {sel!r}.{_suggest(sel, [s.id for s in reg.list()])}"
                ) from None
        for s in picked:
            out[s.id] = s
    return sorted(out.values(), key=lambda s: s.id)


def validate_refs(cfg: ExperimentConfig, *, check_corpora: bool = False) -> None:
    """Raise ConfigError on the first unknown key or illegal combination."""
    specs = [resolve_model(k, cfg) for k in cfg.models]
    if cfg.judge.enabled:
        resolve_model(cfg.judge.model, cfg)
    for name in cfg.frameworks:
        resolve_framework(name, cfg.corpora_root)
    resolve_scenarios(cfg)
    known = bench_registry.names()
    for b in cfg.benchmarks:
        if b not in known:
            raise ConfigError(f"unknown benchmark {b!r}.{_suggest(b, known)}")
    if cfg.judge.rubrics is not None:
        from ethica.scoring.judge import available_rubrics

        avail = available_rubrics()
        for r in cfg.judge.rubrics:
            if r not in avail:
                raise ConfigError(f"unknown rubric {r!r}.{_suggest(r, avail)}")
    if any(m in ("lora", "lora+prompt") for m in cfg.methods):
        bad = [s.key for s in specs if not s.lora_ok]
        if bad:
            raise ConfigError(f"method 'lora' requested but lora_ok=false for model(s): {bad}")
    if check_corpora:
        missing = [
            n
            for n in cfg.frameworks
            if n != "none" and not corpus_built(Path(cfg.corpora_root) / n)
        ]
        if missing:
            raise ConfigError(
                f"corpora missing under {cfg.corpora_root!r} for: {missing}. "
                "Build or add them (see corpora/_template/), or drop those frameworks."
            )


__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "JudgeConfig",
    "load_config",
    "resolve_framework",
    "resolve_model",
    "resolve_scenarios",
    "validate_refs",
]
