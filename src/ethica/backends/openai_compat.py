"""Backend for any OpenAI-compatible local server (llama-server, LM Studio, vllm-mlx)."""

from __future__ import annotations

import time
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from ethica.backends.mock import _truncate_at_stop
from ethica.types import Completion, GenerationParams, Message


class BackendError(RuntimeError):
    pass


class _Msg(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Msg
    finish_reason: str | None = None


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Response(BaseModel):
    choices: list[_Choice]
    usage: _Usage | None = None


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        # LM Studio docs give a base ending in /v1; llama-server docs give the bare host.
        self._url = (
            f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        )
        self.model_id = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # local 7B prompt processing can take minutes: short connect, long read
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0), headers=headers
        )
        self._owns_client = client is None

    def generate(self, messages: list[Message], params: GenerationParams) -> Completion:
        body: dict[str, object] = {
            "model": self.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": False,
        }
        if params.seed is not None:
            body["seed"] = params.seed
        if params.stop:
            body["stop"] = list(params.stop)
        t0 = time.perf_counter()
        try:
            resp = self._client.post(self._url, json=body)
            resp.raise_for_status()
            parsed = _Response.model_validate(resp.json())
        except httpx.HTTPError as e:
            raise BackendError(f"request to {self._url} failed: {e}") from e
        except (ValidationError, ValueError) as e:
            raise BackendError(f"unexpected response shape from {self._url}: {e}") from e
        if not parsed.choices:
            raise BackendError("response has no choices")
        choice = parsed.choices[0]
        text, stopped = _truncate_at_stop(choice.message.content or "", params.stop)
        finish: Literal["stop", "length"] = (
            "length" if choice.finish_reason == "length" and not stopped else "stop"
        )
        usage = parsed.usage or _Usage()
        return Completion(
            text=text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_s=time.perf_counter() - t0,
            finish_reason=finish,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
