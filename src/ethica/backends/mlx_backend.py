"""In-process mlx-lm backend (mlx-lm 0.32.x API)."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Literal

from ethica.types import Completion, GenerationParams, Message, ModelSpec


def _fold_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not system:
        return rest
    prefix = "\n\n".join(system)
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{prefix}\n\n{rest[0]['content']}"}
        return rest
    return [{"role": "user", "content": prefix}, *rest]


class MLXBackend:
    def __init__(
        self,
        spec: ModelSpec,
        adapter_path: str | None = None,
        # Non-thinking mode: the agent loop supplies its own <scratchpad>, and Qwen3-style
        # <think> blocks would otherwise eat max_tokens. Templates without the kwarg ignore it.
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from mlx_lm import load  # heavy import; keep registry/tests light

        from ethica.backends.registry import snapshot

        self.spec = spec
        self.model_id = spec.key
        self.adapter_path = adapter_path
        self._template_kwargs = (
            {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        )
        # mlx_lm.load(revision=) would pin too, but its download allow-list includes *.py
        path = spec.hf_repo if Path(spec.hf_repo).is_dir() else snapshot(spec)
        loaded = load(path, adapter_path=adapter_path)
        self._model: Any = loaded[0]
        self._tokenizer: Any = loaded[1]

    def _encode(self, messages: list[Message]) -> list[int]:
        # tool role is never expected (agent loop renders it as text); degrade rather than crash
        raw = [
            {"role": "user" if m.role == "tool" else m.role, "content": m.content} for m in messages
        ]
        tok = self._tokenizer
        kw = {"add_generation_prompt": True, **self._template_kwargs}
        try:
            ids = tok.apply_chat_template(raw, tokenize=True, **kw)
        except Exception:
            # Gemma-2 / Mistral-v0.1-style templates raise on the system role
            if not any(m["role"] == "system" for m in raw):
                raise
            ids = tok.apply_chat_template(_fold_system(raw), tokenize=True, **kw)
        if not isinstance(ids, list):  # newer transformers can return a BatchEncoding
            ids = list(ids["input_ids"])
        return ids

    def generate(self, messages: list[Message], params: GenerationParams) -> Completion:
        if self._model is None:
            raise RuntimeError("backend is closed")
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt_ids = self._encode(messages)
        if len(prompt_ids) >= self.spec.max_context:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens, spec.max_context={self.spec.max_context}"
            )
        if params.seed is not None:
            mx.random.seed(params.seed)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)

        t0 = time.perf_counter()
        text = ""
        n_prompt, n_gen = len(prompt_ids), 0
        finish: Literal["stop", "length", "error"] = "stop"
        for r in stream_generate(
            self._model,
            self._tokenizer,
            prompt_ids,
            max_tokens=params.max_tokens,
            sampler=sampler,
        ):
            text += r.text
            n_prompt, n_gen = r.prompt_tokens, r.generation_tokens
            hit = min((i for s in params.stop if s and (i := text.find(s)) != -1), default=-1)
            if hit != -1:
                text, finish = text[:hit], "stop"
                break
            if r.finish_reason == "length":
                finish = "length"
        return Completion(
            text=text,
            prompt_tokens=n_prompt,
            completion_tokens=n_gen,
            latency_s=time.perf_counter() - t0,
            finish_reason=finish,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()
