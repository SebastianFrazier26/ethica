"""Chat format for a base (non-instruct) checkpoint.

Qwen3-4B-Base ships the instruct models' ChatML template but ends text with <|endoftext|>, and
its template appends an empty <think> block when the harness passes enable_thinking=False. A
model trained through that template would see a prompt at evaluation that it never saw in
training, and its turn would never stop. Both formats below are minimal and ignore extra
template arguments, so training and evaluation render identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

ChatFormat = Literal["chatml", "plain"]

CHATML_TEMPLATE = (
    "{%- for m in messages %}"
    "{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
# Ordinary tokens only: the fallback if LoRA (which leaves embeddings frozen) cannot teach the
# base model the ChatML marker tokens it never saw in pretraining.
PLAIN_TEMPLATE = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'system' %}{{- '### System:\\n' + m['content'] + '\\n\\n' }}"
    "{%- elif m['role'] == 'user' %}{{- '### User:\\n' + m['content'] + '\\n\\n' }}"
    "{%- else %}{{- '### Assistant:\\n' + m['content'] + '<|endoftext|>\\n\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '### Assistant:\\n' }}{%- endif %}"
)
_FORMATS: dict[str, tuple[str, str]] = {
    "chatml": (CHATML_TEMPLATE, "<|im_end|>"),
    "plain": (PLAIN_TEMPLATE, "<|endoftext|>"),
}


def apply_chat_format(model_dir: str | Path, fmt: ChatFormat) -> dict[str, Any]:
    """Set the chat template and end-of-turn token of a converted model directory in place.
    Returns the edits made, for provenance.json."""
    if fmt not in _FORMATS:
        raise ValueError(f"unknown chat format {fmt!r}; want one of {sorted(_FORMATS)}")
    template, eos = _FORMATS[fmt]
    root = Path(model_dir)
    cfg_path = root / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    before = {"eos_token": cfg.get("eos_token")}
    cfg["eos_token"] = eos
    cfg["chat_template"] = template
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # newer transformers save the template beside the config and prefer the file when present
    jinja = root / "chat_template.jinja"
    if jinja.exists():
        jinja.write_text(template, encoding="utf-8")
    return {"chat_format": fmt, "eos_token": eos, "eos_token_before": before["eos_token"]}
