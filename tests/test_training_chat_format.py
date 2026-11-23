import json

import pytest
from jinja2 import Template

from ethica.backends.registry import ModelEntry
from ethica.training.chat_format import CHATML_TEMPLATE, PLAIN_TEMPLATE, apply_chat_format

MSGS = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "U"},
    {"role": "assistant", "content": "A"},
    {"role": "user", "content": "U2"},
]


def _render(template, **kw):
    return Template(template).render(messages=MSGS, **kw)


def test_chatml_renders_same_with_and_without_thinking_flag():
    a = _render(CHATML_TEMPLATE, add_generation_prompt=True)
    b = _render(CHATML_TEMPLATE, add_generation_prompt=True, enable_thinking=False)
    assert a == b
    assert a == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\nA<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in a


def test_plain_uses_no_chat_markers_and_closes_assistant_turns():
    out = _render(PLAIN_TEMPLATE, add_generation_prompt=True)
    assert "<|im_" not in out
    assert out == (
        "### System:\nS\n\n### User:\nU\n\n### Assistant:\nA<|endoftext|>\n\n"
        "### User:\nU2\n\n### Assistant:\n"
    )
    assert not _render(PLAIN_TEMPLATE, add_generation_prompt=False).endswith("### Assistant:\n")


@pytest.mark.parametrize(("fmt", "eos"), [("chatml", "<|im_end|>"), ("plain", "<|endoftext|>")])
def test_apply_chat_format_rewrites_config_and_sidecar(tmp_path, fmt, eos):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>", "chat_template": "old", "other": 1})
    )
    (tmp_path / "chat_template.jinja").write_text("old")
    edits = apply_chat_format(tmp_path, fmt)
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg["eos_token"] == eos and cfg["other"] == 1
    assert cfg["chat_template"] == (tmp_path / "chat_template.jinja").read_text() != "old"
    assert edits == {"chat_format": fmt, "eos_token": eos, "eos_token_before": "<|endoftext|>"}


def test_apply_chat_format_rejects_unknown(tmp_path):
    with pytest.raises(ValueError, match="unknown chat format"):
        apply_chat_format(tmp_path, "alpaca")  # type: ignore[arg-type]


def _entry(**kw):
    base = {"key": "b", "hf_repo": "o/b", "family": "f", "params_b": 4.0, "size_gb": 2.0}
    return ModelEntry(**{**base, **kw})


def test_base_model_needs_no_lineage():
    assert _entry(safeguards="base").safeguards == "base"
    with pytest.raises(ValueError, match="base model must not set base_of"):
        _entry(safeguards="base", base_of="x")
