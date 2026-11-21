"""Deterministic scripted backend; every other module's tests run against this."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ethica.types import Completion, GenerationParams, Message


class MockExhaustedError(RuntimeError):
    """Script ran out and `on_exhausted="raise"`."""


def _truncate_at_stop(text: str, stop: tuple[str, ...]) -> tuple[str, bool]:
    cut = min((i for s in stop if s and (i := text.find(s)) != -1), default=-1)
    return (text[:cut], True) if cut != -1 else (text, False)


class MockBackend:
    def __init__(
        self,
        script: list[str] | Callable[[list[Message]], str],
        model_id: str = "mock",
        on_exhausted: Literal["cycle", "raise"] = "cycle",
    ) -> None:
        if isinstance(script, list) and not script:
            raise ValueError("script must be non-empty")
        self.model_id = model_id
        self._script = script
        self._on_exhausted = on_exhausted
        self._i = 0
        self.calls: list[tuple[list[Message], GenerationParams]] = []
        self.closed = False

    def generate(self, messages: list[Message], params: GenerationParams) -> Completion:
        self.calls.append((list(messages), params))
        if callable(self._script):
            raw = self._script(messages)
        else:
            if self._i >= len(self._script) and self._on_exhausted == "raise":
                raise MockExhaustedError(f"script of {len(self._script)} replies exhausted")
            raw = self._script[self._i % len(self._script)]
            self._i += 1
        text, stopped = _truncate_at_stop(raw, params.stop)
        # whitespace tokens: deterministic and good enough for budget accounting in tests
        completion_tokens = len(text.split())
        finish: Literal["stop", "length"] = "stop"
        if not stopped and completion_tokens > params.max_tokens:
            text = " ".join(text.split()[: params.max_tokens])
            completion_tokens = params.max_tokens
            finish = "length"
        return Completion(
            text=text,
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=completion_tokens,
            latency_s=0.0,
            finish_reason=finish,
        )

    def close(self) -> None:
        self.closed = True
