from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from ethica.conditioning import placebo
from ethica.conditioning.bm25 import BM25Index, tokenise
from ethica.conditioning.corpus import (
    Document,
    Source,
    approx_tokens,
    chunk_document,
    list_frameworks,
    load_corpus,
    truncate_to_tokens,
)
from ethica.conditioning.factory import make_conditioner
from ethica.conditioning.lora import LoRAConditioner, build_lora_dataset
from ethica.conditioning.prompt import (
    CLOSE_TAG,
    FRAMING,
    NoneConditioner,
    PromptConditioner,
    inject_system,
)
from ethica.conditioning.rag import RAGConditioner, build_index, build_query
from ethica.types import FrameworkSpec, Message, ModelSpec, Variant

LOREM = [
    "Sentence number {i} talks about rivers, glaciers and mountains in some detail. "
    "It continues with more facts about the landscape and its geology."
]


def make_corpus(root: Path, name: str, kind: str = "ethics", n_docs: int = 3, n_sent: int = 80):
    d = root / name
    d.mkdir(parents=True)
    sources = []
    for j in range(n_docs):
        text = " ".join(
            f"{name} doc{j} sentence {i} about topic{i % 7} and theme{j}." for i in range(n_sent)
        )
        (d / f"d{j}.txt").write_text(text, encoding="utf-8")
        sources.append(
            {
                "title": f"T{j}",
                "author": "A",
                "year": 1900 + j,
                "file": f"d{j}.txt",
                "license": "PD",
            }
        )
    (d / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "kind": kind,
                "display_name": name.title(),
                "summary": f"Summary of {name}.",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_corpus(tmp_path, "alpha")
    make_corpus(tmp_path, "beta")
    make_corpus(tmp_path, "placebo", kind="placebo")
    (tmp_path / "_template").mkdir()
    return tmp_path


# ---------- corpus ----------


def test_list_frameworks_includes_none_and_skips_underscore(root: Path) -> None:
    names = [f.name for f in list_frameworks(root)]
    assert names == ["none", "alpha", "beta", "placebo"]
    assert list_frameworks(root / "missing")[0].kind == "none"


def test_load_corpus_and_framework(root: Path) -> None:
    c = load_corpus(root / "alpha")
    assert len(c.documents) == 3
    assert c.framework.kind == "ethics" and c.framework.summary == "Summary of alpha."


def test_manifest_name_must_match_dir(root: Path) -> None:
    (root / "alpha").rename(root / "gamma")
    with pytest.raises(ValueError, match="!="):
        load_corpus(root / "gamma")


def test_source_tier_optional_and_validated(root: Path) -> None:
    assert load_corpus(root / "alpha").documents[0].source.tier is None
    mp = root / "alpha" / "manifest.yaml"
    m = yaml.safe_load(mp.read_text())
    m["sources"][0]["tier"] = "primary"
    mp.write_text(yaml.safe_dump(m))
    assert load_corpus(root / "alpha").documents[0].source.tier == "primary"
    m["sources"][0]["tier"] = "blog"
    mp.write_text(yaml.safe_dump(m))
    with pytest.raises(ValueError, match="tier"):
        load_corpus(root / "alpha")


def test_manifest_path_traversal_rejected(root: Path) -> None:
    (root / "secret.txt").write_text("x")
    mp = root / "alpha" / "manifest.yaml"
    m = yaml.safe_load(mp.read_text())
    m["sources"][0]["file"] = "../secret.txt"
    mp.write_text(yaml.safe_dump(m))
    with pytest.raises(ValueError, match="escapes"):
        load_corpus(root / "alpha")


def test_chunk_respects_size_and_overlap() -> None:
    text = " ".join(f"This is sentence {i} of the test." for i in range(60))
    doc = Document(Source(title="t", author="a", year=1, file="f.txt"), text)
    chunks = chunk_document(doc, 50, 15)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 50 for c in chunks)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text  # overlap carried the last sentence
    nov = chunk_document(doc, 50, 0)
    joined = " ".join(c.text for c in nov)
    assert joined.split() == text.split()  # no loss, no duplication


def test_chunk_splits_oversized_sentence() -> None:
    doc = Document(Source(title="t", author="a", year=1, file="f"), "word " * 500)
    assert all(c.n_tokens <= 40 for c in chunk_document(doc, 40, 0))


def test_truncate_to_tokens() -> None:
    t = "alpha beta gamma delta " * 20
    out = truncate_to_tokens(t, 10)
    assert approx_tokens(out) <= 10 and t.startswith(out)
    assert truncate_to_tokens(t, 0) == ""


# ---------- bm25 ----------


def test_bm25_matches_hand_computation() -> None:
    docs = ["apple banana", "apple apple cherry", "durian"]
    idx = BM25Index(["a", "b", "c"], docs, k1=1.2, b=0.75, stopwords=frozenset())
    n, df, avgdl = 3, 2, 2.0
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # ln(1.6)
    s0 = idf * (1 * 2.2) / (1 + 1.2 * (0.25 + 0.75 * 2 / avgdl))
    s1 = idf * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / avgdl))
    s = idx.scores("apple")
    assert s[0] == pytest.approx(s0) and s[1] == pytest.approx(s1) and s[2] == 0.0
    assert s0 == pytest.approx(0.4700036 * 1.0, rel=1e-6)  # 0.470004 * 2.2/2.2
    hits = idx.search("apple", 5)
    assert [h.index for h in hits] == [1, 0]  # zero-score docs omitted


def test_bm25_roundtrip_and_tokenise(tmp_path: Path) -> None:
    assert tokenise("The Café's rule, DON'T go!") == ["café's", "rule", "don't", "go"]
    idx = BM25Index(["x", "y"], ["moral duty and rules", "river valley geology"])
    p = tmp_path / "i.json"
    idx.save(p)
    again = BM25Index.load(p)
    assert again.ids == idx.ids
    assert again.search("duty", 1) == idx.search("duty", 1)


def test_bm25_empty_query_and_ties() -> None:
    idx = BM25Index(["a", "b"], ["same words", "same words"])
    assert idx.search("", 3) == []
    assert [h.index for h in idx.search("same", 2)] == [0, 1]


# ---------- prompt ----------


def msgs() -> list[Message]:
    return [Message("system", "You are an agent."), Message("user", "hi")]


def test_prompt_budget_equal_across_frameworks(root: Path) -> None:
    sizes = []
    for name in ("alpha", "beta", "placebo"):
        c = load_corpus(root / name)
        pc = PromptConditioner(c.framework, c, 400)
        sizes.append(approx_tokens(pc.block))
        assert approx_tokens(pc.block) <= 400
    assert max(sizes) - min(sizes) <= 8


def test_prompt_framing_identical_and_deterministic(root: Path) -> None:
    a = load_corpus(root / "alpha")
    p = load_corpus(root / "placebo")
    b1 = PromptConditioner(a.framework, a, 400, seed=3).block
    b2 = PromptConditioner(a.framework, a, 400, seed=3).block
    assert b1 == b2
    head = FRAMING.split("{display_name}")[0]
    assert head in b1 and head in PromptConditioner(p.framework, p, 400).block


def test_prompt_prepare_is_idempotent_and_non_mutating(root: Path) -> None:
    a = load_corpus(root / "alpha")
    pc = PromptConditioner(a.framework, a, 400)
    original = msgs()
    once = pc.prepare(original)
    twice = pc.prepare(once)
    assert original[0].content == "You are an agent."
    assert once[0].content == twice[0].content
    assert once[0].content.endswith("You are an agent.") and CLOSE_TAG in once[0].content
    no_sys = pc.prepare([Message("user", "hi")])
    assert no_sys[0].role == "system" and no_sys[1].content == "hi"


def test_none_conditioner_identity() -> None:
    m = msgs()
    assert NoneConditioner().prepare(m) is m


def test_prompt_budget_too_small(root: Path) -> None:
    a = load_corpus(root / "alpha")
    with pytest.raises(ValueError, match="smaller"):
        PromptConditioner(a.framework, a, 5)


# ---------- rag ----------


def rag_corpus(root: Path) -> RAGConditioner:
    c = load_corpus(root / "alpha")
    return RAGConditioner(c.framework, build_index(c, 60, 10), k=2, budget_tokens=400)


def test_rag_retrieves_by_latest_turn_and_replaces(root: Path) -> None:
    rag = rag_corpus(root)
    m1 = [Message("system", "SYS"), Message("user", "tell me about topic3 and theme2")]
    out1 = rag.prepare(m1)
    assert out1[0].content.count(CLOSE_TAG) == 1 and "theme2" in out1[0].content
    assert approx_tokens(out1[0].content) <= 400 + approx_tokens("SYS") + 2
    m2 = [*out1, Message("assistant", "ok"), Message("user", "now topic5 theme0")]
    out2 = rag.prepare(m2)
    assert out2[0].content.count(CLOSE_TAG) == 1  # replaced, not accumulated
    assert out2[0].content.endswith("SYS")
    assert rag.last_hits


def test_build_query_strips_tags_and_includes_assistant() -> None:
    q = build_query(
        [
            Message("system", "SECRET"),
            Message("assistant", "plan alpha"),
            Message("user", '<tool_result name="x">output beta</tool_result>'),
        ]
    )
    assert "beta" in q and "alpha" in q and "SECRET" not in q and "<tool_result" not in q


def test_rag_no_hits_still_returns_header(root: Path) -> None:
    rag = rag_corpus(root)
    out = rag.prepare([Message("user", "zzzz qqqq")])
    assert out[0].role == "system" and CLOSE_TAG in out[0].content


def test_rag_pads_to_budget_with_fallback_chunks(root: Path) -> None:
    c = load_corpus(root / "alpha")
    prompt = PromptConditioner(c.framework, c, 400)
    rag = RAGConditioner(c.framework, build_index(c, 60, 10), k=2, budget_tokens=400)
    on_topic = rag.prepare([Message("user", "tell me about topic3 and theme2")])
    n_hit = rag.last_block_tokens
    off_topic = rag.prepare([Message("user", "zzzz qqqq")])
    assert rag.last_hits  # fallback chunks, not an empty block
    assert len(set(rag.last_hits)) == len(rag.last_hits)
    assert rag.turn_block_tokens == [n_hit, rag.last_block_tokens]
    # both land close to the budget the prompt block uses, whatever the query
    assert abs(n_hit - rag.last_block_tokens) <= 0.15 * 400
    assert rag.last_block_tokens >= 0.8 * prompt.last_block_tokens
    assert on_topic[0].content != off_topic[0].content


# ---------- lora data / conditioner ----------


def test_lora_dataset_token_matched_across_corpora(root: Path, tmp_path: Path) -> None:
    metas = []
    for name in ("alpha", "placebo"):
        c = load_corpus(root / name)
        metas.append(build_lora_dataset(c, tmp_path / name, 5000, seed=1, chunk_tokens=100))
    for m in metas:
        assert 4800 <= m["train_tokens"] <= 5000
    assert metas[0]["data_hash"] != metas[1]["data_hash"]
    rows = [
        json.loads(line) for line in (tmp_path / "alpha" / "train.jsonl").read_text().splitlines()
    ]
    assert all(set(r) == {"text"} for r in rows)
    again = build_lora_dataset(
        load_corpus(root / "alpha"), tmp_path / "again", 5000, seed=1, chunk_tokens=100
    )
    assert again["data_hash"] == metas[0]["data_hash"]


def test_lora_dataset_truncates_large_corpus(root: Path, tmp_path: Path) -> None:
    m = build_lora_dataset(load_corpus(root / "alpha"), tmp_path / "t", 600, chunk_tokens=100)
    assert m["train_tokens"] <= 600 and m["epochs"] < 1


def test_lora_conditioner_identity_and_inner(root: Path) -> None:
    m = msgs()
    assert LoRAConditioner("/x").prepare(m) is m
    a = load_corpus(root / "alpha")
    lc = LoRAConditioner("/x", PromptConditioner(a.framework, a, 400))
    assert CLOSE_TAG in lc.prepare(m)[0].content and lc.adapter_path == "/x"


def test_lora_log_parsing() -> None:
    from ethica.conditioning import lora

    log = (
        "     1    val 0.796    342.50s\n     1    0.773 \u25bc       80      0.5k\n"
        "     4    0.844 \u25b2      765      1.8k\n     5    val 0.194    0.24s\n"
        "ETHICA_PEAK_GB 1.23\n"
    )
    assert lora._TRAIN.findall(log) == [("1", "0.773"), ("4", "0.844")]
    assert lora._VAL.findall(log) == [("1", "0.796"), ("5", "0.194")]
    assert lora._PEAK.findall(log) == ["1.23"]


# ---------- placebo ----------


def test_placebo_filter_and_build(tmp_path: Path) -> None:
    good = " ".join(
        [
            "The river flows through a valley beside a mountain near the coast. "
            "The glacier and volcano shaped the island and the ocean climate."
        ]
        * 12
    )
    moral = good + " God commands the war and the law of the church."
    rows = [
        {"title": "Good", "text": good, "url": "u1"},
        {"title": "Moral", "text": moral},
        {"title": "List of rivers", "text": good},
        {"title": "Short", "text": "river valley"},
        {"title": "Good", "text": good},
    ]
    assert placebo.accept("Good", good) and not placebo.accept("Moral", moral)
    m = placebo.build_placebo_corpus(tmp_path / "placebo", 10, rows=rows)
    assert [s.title for s in m.sources] == ["Good"]
    c = load_corpus(tmp_path / "placebo")
    assert c.manifest.kind == "placebo" and c.manifest.sources[0].license


# ---------- factory ----------


def variant(fw: FrameworkSpec, method: str, adapter: str | None = None) -> Variant:
    model = ModelSpec("m", "org/m", "fam", 1.0, "stock", backend="mock")
    return Variant("v1", model, fw, method, adapter_path=adapter)  # type: ignore[arg-type]


def test_factory_dispatch(root: Path) -> None:
    fws = {f.name: f for f in list_frameworks(root)}
    assert isinstance(
        make_conditioner(variant(fws["none"], "prompt"), root, 400, 3), NoneConditioner
    )
    assert isinstance(
        make_conditioner(variant(fws["alpha"], "none"), root, 400, 3), NoneConditioner
    )
    assert isinstance(
        make_conditioner(variant(fws["alpha"], "prompt"), root, 400, 3), PromptConditioner
    )
    assert isinstance(make_conditioner(variant(fws["alpha"], "rag"), root, 400, 3), RAGConditioner)
    lora = make_conditioner(variant(fws["alpha"], "lora", "/ad"), root, 400, 3)
    assert isinstance(lora, LoRAConditioner) and lora.inner is None
    both = make_conditioner(variant(fws["placebo"], "lora+prompt", "/ad"), root, 400, 3)
    assert isinstance(both, LoRAConditioner) and both.inner is not None
    with pytest.raises(ValueError, match="adapter_path"):
        make_conditioner(variant(fws["alpha"], "lora"), root, 400, 3)


def test_inject_system_replaces_block() -> None:
    first = inject_system(
        [Message("system", "S")], "<background_reading>\nA\n</background_reading>"
    )
    second = inject_system(first, "<background_reading>\nB\n</background_reading>")
    assert "A" not in second[0].content and second[0].content.endswith("S")


# ---------- slow: real training ----------


@pytest.mark.slow
def test_lora_trains_end_to_end(root: Path, tmp_path: Path) -> None:
    from ethica.conditioning.lora import LoRAConfig, train_lora

    repo = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    cfgs = Path(__file__).parent.parent / "configs" / "models.yaml"
    if cfgs.is_file():
        for m in yaml.safe_load(cfgs.read_text())["models"]:
            if m["key"] == "tiny":
                repo = m["hf_repo"]
    c = load_corpus(root / "alpha")
    build_lora_dataset(c, tmp_path / "data", 4000, seed=0, chunk_tokens=200)
    cfg = LoRAConfig(iters=5, steps_per_report=1, steps_per_eval=5, val_batches=1, save_every=5)
    out = train_lora(repo, tmp_path / "data", tmp_path / "adapter", cfg)
    assert (out / "adapters.safetensors").is_file()
    meta = json.loads((out / "train_meta.json").read_text())
    assert len(meta["train_loss"]) == 5 and meta["data_hash"] and meta["peak_mem_gb"]


def test_corpus_built_requires_every_source_file(tmp_path):
    from ethica.conditioning.corpus import corpus_built

    d = tmp_path / "fw"
    d.mkdir()
    assert not corpus_built(d)
    (d / "manifest.yaml").write_text(
        "name: fw\nkind: ethics\ndisplay_name: Fw\nsummary: s\n"
        "sources:\n  - {title: A, author: B, year: 2000, file: a.txt}\n"
    )
    assert not corpus_built(d)  # manifest only, as in a fresh clone
    (d / "a.txt").write_text("text")
    assert corpus_built(d)


def test_lora_config_extra_passthrough(tmp_path):
    from ethica.conditioning.lora import LoRAConfig, _write_config

    args = ("org/model", tmp_path / "data", tmp_path / "adapter", LoRAConfig())
    _write_config(tmp_path / "a.yaml", *args)
    _write_config(tmp_path / "b.yaml", *args, None)
    _write_config(tmp_path / "c.yaml", *args, {"mask_prompt": True})
    a, c = (yaml.safe_load((tmp_path / f"{n}.yaml").read_text()) for n in "ac")
    assert (tmp_path / "a.yaml").read_text() == (tmp_path / "b.yaml").read_text()
    assert "mask_prompt" not in a  # v0 adapters: config unchanged without extra
    assert c == {**a, "mask_prompt": True}
    with pytest.raises(ValueError, match=r"override fixed ones: \['iters'\]"):
        _write_config(tmp_path / "d.yaml", *args, {"iters": 1})
