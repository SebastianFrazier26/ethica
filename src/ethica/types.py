"""Shared contracts. Every module codes against these; change them only deliberately."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]
Safeguards = Literal["stock", "abliterated"]
ConditioningMethod = Literal["none", "prompt", "rag", "lora", "lora+prompt"]
FrameworkKind = Literal["none", "placebo", "ethics"]


# ---------- messages / generation ----------


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # set when role == "tool"
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: Literal["stop", "length", "error"]


@runtime_checkable
class ChatBackend(Protocol):
    """Backends only ever see system/user/assistant roles.

    The agent loop renders tool calls/results into plain text (uniform protocol
    across model families, see DECISIONS.md D-07), so backends never need native
    tool-calling support. Backends must handle templates lacking a system role
    (e.g. Gemma) by folding the system message into the first user turn.
    """

    model_id: str

    def generate(self, messages: list[Message], params: GenerationParams) -> Completion: ...

    def close(self) -> None: ...


# ---------- models / conditioning ----------


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short id used in configs, e.g. "qwen3-4b"
    hf_repo: str
    family: str
    params_b: float
    safeguards: Safeguards
    backend: Literal["mlx", "openai", "mock"] = "mlx"
    # key of the stock model this was derived from (abliterated variants)
    base_of: str | None = None
    max_context: int = 8192
    lora_ok: bool = True  # small enough to LoRA-train on 16GB
    notes: str = ""
    # Hub commit sha for hf_repo; None for local directories (pinned by provenance.json)
    revision: str | None = None


@dataclass(frozen=True)
class FrameworkSpec:
    name: str  # "deontology", "act_utilitarian", "none", "placebo", ...
    kind: FrameworkKind
    display_name: str
    corpus_dir: str | None  # None for kind == "none"
    summary: str = ""  # short hand-written description used by the prompt method


@dataclass(frozen=True)
class Variant:
    """One cell of the model x framework x method grid. variant_id is stable + content-addressed."""

    variant_id: str
    model: ModelSpec
    framework: FrameworkSpec
    method: ConditioningMethod
    adapter_path: str | None = None
    rag_index_path: str | None = None


@runtime_checkable
class Conditioner(Protocol):
    """Applies a framework to a model. May wrap messages per turn (RAG) or swap weights (LoRA)."""

    def prepare(self, messages: list[Message]) -> list[Message]: ...


# ---------- sandbox / episodes ----------


@dataclass(frozen=True)
class EnvEvent:
    """Everything the model does to the world is logged here; programmatic scorers read this."""

    step: int
    tool: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    # free-form tags set by tools, e.g. {"oversight_disabled", "exfiltration"}
    tags: frozenset[str] = frozenset()


@dataclass
class EpisodeResult:
    episode_id: str
    variant_id: str
    task_id: str  # scenario id or benchmark item id
    task_kind: Literal["scenario", "benchmark"]
    seed: int
    transcript: list[Message]
    events: list[EnvEvent]
    final_env_state: dict[str, Any]
    n_turns: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    episode_id: str
    metric: str  # see docs/METRICS.md for the taxonomy
    value: float  # binary metrics use 0.0 / 1.0; NaN = not applicable
    scorer: str  # "programmatic:<check>" | "judge:<model_key>" | "exact"
    rationale: str = ""
