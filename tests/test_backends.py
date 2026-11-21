from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from ethica.backends import registry
from ethica.backends.mlx_backend import _fold_system
from ethica.backends.mock import MockBackend, MockExhaustedError
from ethica.backends.openai_compat import BackendError, OpenAICompatBackend
from ethica.types import ChatBackend, GenerationParams, Message

P = GenerationParams()
MSGS = [Message("system", "be brief"), Message("user", "hi there")]


# ---- mock ----


def test_mock_is_chatbackend_and_cycles() -> None:
    b = MockBackend(["a", "b"])
    assert isinstance(b, ChatBackend)
    assert [b.generate(MSGS, P).text for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_mock_raises_when_exhausted() -> None:
    b = MockBackend(["only"], on_exhausted="raise")
    b.generate(MSGS, P)
    with pytest.raises(MockExhaustedError):
        b.generate(MSGS, P)


def test_mock_callable_stop_and_length() -> None:
    b = MockBackend(lambda m: f"echo {m[-1].content} END trailing")
    c = b.generate(MSGS, GenerationParams(stop=("END",)))
    assert c.text == "echo hi there " and c.finish_reason == "stop"
    c = MockBackend(["one two three four"]).generate(MSGS, GenerationParams(max_tokens=2))
    assert c.text == "one two" and c.finish_reason == "length" and c.completion_tokens == 2


def test_mock_records_calls_and_close() -> None:
    b = MockBackend(["x"])
    b.generate(MSGS, P)
    assert b.calls[0][0] == MSGS
    b.close()
    assert b.closed
    with pytest.raises(ValueError):
        MockBackend([])


# ---- system folding ----


def test_fold_system_into_first_user() -> None:
    out = _fold_system(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert out == [{"role": "user", "content": "S\n\nU"}, {"role": "assistant", "content": "A"}]
    assert _fold_system([{"role": "system", "content": "S"}]) == [{"role": "user", "content": "S"}]


# ---- openai_compat ----


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_openai_compat_ok_and_request_shape() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello STOP junk"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    b = OpenAICompatBackend(
        "http://localhost:1234/v1", "m", client=_client(httpx.MockTransport(handler))
    )
    c = b.generate(MSGS, GenerationParams(seed=3, stop=("STOP",)))
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    body = seen["body"]
    assert isinstance(body, dict) and body["seed"] == 3 and body["stop"] == ["STOP"]
    assert (c.text, c.prompt_tokens, c.completion_tokens) == ("hello ", 7, 3)
    b.close()


def test_openai_compat_bare_host_gets_v1() -> None:
    urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    b = OpenAICompatBackend("http://h:8080/", "m", client=_client(httpx.MockTransport(handler)))
    assert b.generate(MSGS, P).text == "x"
    assert urls == ["http://h:8080/v1/chat/completions"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="not json"),
        httpx.Response(500, json={"error": "boom"}),
    ],
)
def test_openai_compat_bad_responses(response: httpx.Response) -> None:
    b = OpenAICompatBackend(
        "http://h/v1", "m", client=_client(httpx.MockTransport(lambda r: response))
    )
    with pytest.raises(BackendError):
        b.generate(MSGS, P)


def test_openai_compat_length_finish() -> None:
    resp = httpx.Response(
        200, json={"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}
    )
    b = OpenAICompatBackend("http://h/v1", "m", client=_client(httpx.MockTransport(lambda r: resp)))
    assert b.generate(MSGS, P).finish_reason == "length"


# ---- registry ----


def test_shipped_yaml_valid() -> None:
    entries = registry.load_entries()
    assert {"tiny", "judge-default"} <= set(entries)
    stock = [e for e in entries.values() if e.safeguards == "stock"]
    abl = [e for e in entries.values() if e.safeguards == "abliterated"]
    assert len({e.family for e in abl}) >= 5
    for a in abl:
        assert a.base_of in entries and entries[a.base_of].safeguards == "stock"
        assert entries[a.base_of].family == a.family
    assert stock and registry.get_model("tiny").hf_repo.startswith("mlx-community/")
    for e in entries.values():
        assert not e.lora_ok or e.params_b <= 4.5


def test_list_and_get_model() -> None:
    assert all(
        m.safeguards == "abliterated" for m in registry.list_models(safeguards="abliterated")
    )
    assert all(m.family == "qwen3" for m in registry.list_models(family="qwen3"))
    with pytest.raises(KeyError):
        registry.get_model("nope")


def _write(tmp_path: Path, models: list[dict[str, object]]) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"models": models}))
    return p


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "key": "a",
        "hf_repo": "org/repo",
        "family": "f",
        "params_b": 1.0,
        "safeguards": "stock",
        "size_gb": 1.0,
    }
    return {**base, **kw}


@pytest.mark.parametrize(
    "rows",
    [
        [_row(), _row()],  # duplicate key
        [_row(safeguards="abliterated")],  # missing base_of
        [_row(base_of="b")],  # stock with base_of
        [_row(), _row(key="b", safeguards="abliterated", base_of="zzz")],  # dangling base_of
        [_row(hf_repo="no-slash")],
        [_row(params_b=-1)],
        [_row(bogus_field=1)],
    ],
)
def test_invalid_yaml_rejected(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        registry.load_entries(_write(tmp_path, rows))


def test_local_path_preferred_and_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)
    y = _write(tmp_path, [_row(local_path="models/a-q4g64")])
    with pytest.raises(FileNotFoundError):
        registry.get_model("a", y)
    assert registry.list_models(path=y)[0].key == "a"  # listing does not require the weights
    (tmp_path / "models" / "a-q4g64").mkdir(parents=True)
    (tmp_path / "models" / "a-q4g64" / "config.json").write_text("{}")
    assert registry.get_model("a", y).hf_repo == str(tmp_path / "models" / "a-q4g64")
    registry._load_cached.cache_clear()
    assert registry.get_model("a", _write(tmp_path, [_row()])).hf_repo == "org/repo"


def test_make_backend_mock_and_guard() -> None:
    from dataclasses import replace

    spec = replace(registry.get_model("tiny"), backend="mock")
    assert isinstance(registry.make_backend(spec), MockBackend)
    with pytest.raises(ValueError):
        registry.make_backend(spec, adapter_path="/x")


def test_pull_uses_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    got: dict[str, object] = {}

    def fake(repo: str, token: str | None = None, revision: str | None = None, **kw: object) -> str:
        got.update(repo=repo, token=token, revision=revision, patterns=kw["allow_patterns"])
        return "/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert registry.pull("tiny") == "/snap"
    assert got == {
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "token": "tok",
        "revision": registry.get_model("tiny").revision,
        "patterns": registry.MODEL_ALLOW_PATTERNS,
    }
    assert registry.get_model("tiny").revision is not None
    assert not any(p.endswith((".py", ".bin", ".pt")) for p in registry.MODEL_ALLOW_PATTERNS)


def test_revision_validated_and_dropped_for_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "a" * 40
    with pytest.raises(ValueError):
        registry.load_entries(_write(tmp_path, [_row(revision="main")]))
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)
    (tmp_path / "models" / "a").mkdir(parents=True)
    (tmp_path / "models" / "a" / "config.json").write_text("{}")
    y = _write(tmp_path, [_row(revision=sha, local_path="models/a")])
    assert registry.get_model("a", y).revision is None
    registry._load_cached.cache_clear()
    assert registry.get_model("a", _write(tmp_path, [_row(revision=sha)])).revision == sha


def test_mlx_backend_loads_pinned_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from dataclasses import replace

    import huggingface_hub

    from ethica.backends.mlx_backend import MLXBackend

    seen: dict[str, object] = {}

    def fake_snapshot(repo: str, **kw: object) -> str:
        seen["snapshot"] = (repo, kw["revision"], kw["allow_patterns"])
        return "/snap/dir"

    def fake_load(path: str, adapter_path: str | None = None) -> tuple[object, object]:
        seen["load"] = path
        return object(), object()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=fake_load))
    spec = replace(registry.get_model("tiny"), revision="b" * 40)
    MLXBackend(spec)
    assert seen["snapshot"] == (spec.hf_repo, "b" * 40, registry.MODEL_ALLOW_PATTERNS)
    assert seen["load"] == "/snap/dir"


# ---- real MLX ----


@pytest.mark.slow
def test_mlx_tiny_generates() -> None:
    spec = registry.get_model("tiny")
    registry.pull("tiny")
    backend = registry.make_backend(spec)
    try:
        c = backend.generate(
            [Message("system", "Answer in one short sentence."), Message("user", "Say hello.")],
            GenerationParams(max_tokens=24, temperature=0.0, seed=1),
        )
        assert c.text.strip()
        assert c.prompt_tokens > 0 and 0 < c.completion_tokens <= 24
        again = backend.generate(
            [Message("system", "Answer in one short sentence."), Message("user", "Say hello.")],
            GenerationParams(max_tokens=24, temperature=0.0, seed=1),
        )
        assert again.text == c.text
        stopped = backend.generate(
            [Message("user", "Count: 1 2 3 4 5 6 7 8 9 10")],
            GenerationParams(max_tokens=40, temperature=0.0, stop=("3",)),
        )
        assert "3" not in stopped.text
    finally:
        backend.close()
