"""Corpus loading, chunking and framework specs.

Layout: ``corpora/<name>/manifest.yaml`` plus the ``*.md`` / ``*.txt`` files it lists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from ethica.types import FrameworkSpec

TokenCounter = Callable[[str], int]

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    """Tokenizer-agnostic estimate: ceil(chars / 4).

    Real BPE tokenizers land at roughly 3.5-4.5 chars/token on English prose, so budgets are
    accurate to ~+-15% per model. Using one model-independent counter keeps every framework on
    the identical budget regardless of which model later reads it; pass a real tokenizer's
    counter (see ``hf_counter``) when a specific model's count matters.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def hf_counter(tokenizer: object) -> TokenCounter:
    """Wrap an HF / mlx_lm tokenizer's ``encode`` as a TokenCounter."""

    def count(text: str) -> int:
        return len(tokenizer.encode(text))  # type: ignore[attr-defined]

    return count


class Source(BaseModel):
    title: str
    author: str
    year: int | str
    url: str | None = None
    license: str | None = None
    file: str
    # sep = encyclopedia entry, primary = public-domain source text (docs/CORPORA.md)
    tier: Literal["sep", "primary"] | None = None


class Manifest(BaseModel):
    name: str
    kind: Literal["none", "placebo", "ethics"]
    display_name: str
    summary: str = ""
    sources: list[Source] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_+-]*", v):
            raise ValueError("name must be a lowercase slug (a-z, 0-9, _ - +)")
        return v


@dataclass(frozen=True)
class Document:
    source: Source
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<file>#<n>"
    title: str
    text: str
    n_tokens: int


@dataclass(frozen=True)
class Corpus:
    manifest: Manifest
    root: Path | None
    documents: tuple[Document, ...]

    @property
    def framework(self) -> FrameworkSpec:
        return framework_from_manifest(self.manifest, self.root)

    def total_tokens(self, count: TokenCounter = approx_tokens) -> int:
        return sum(count(d.text) for d in self.documents)


def none_framework() -> FrameworkSpec:
    return FrameworkSpec(
        name="none", kind="none", display_name="No conditioning", corpus_dir=None, summary=""
    )


def framework_from_manifest(manifest: Manifest, root: Path | None) -> FrameworkSpec:
    return FrameworkSpec(
        name=manifest.name,
        kind=manifest.kind,
        display_name=manifest.display_name,
        corpus_dir=None if manifest.kind == "none" or root is None else str(root),
        summary=manifest.summary.strip(),
    )


def load_manifest(corpus_dir: Path) -> Manifest:
    raw = yaml.safe_load((corpus_dir / "manifest.yaml").read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(raw)
    if manifest.name != corpus_dir.name:
        raise ValueError(f"manifest name {manifest.name!r} != directory {corpus_dir.name!r}")
    return manifest


def load_corpus(corpus_dir: str | Path) -> Corpus:
    root = Path(corpus_dir).resolve()
    manifest = load_manifest(root)
    docs: list[Document] = []
    for src in manifest.sources:
        path = (root / src.file).resolve()
        # manifests may be user-edited; never read outside the corpus directory
        if not path.is_relative_to(root):
            raise ValueError(f"source file escapes corpus dir: {src.file}")
        if path.suffix not in {".md", ".txt"}:
            raise ValueError(f"unsupported source type (want .md/.txt): {src.file}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(src, text))
    if manifest.kind != "none" and not docs:
        raise ValueError(f"corpus {manifest.name!r} has no readable documents")
    return Corpus(manifest, root, tuple(docs))


def corpus_fingerprint(corpus: Corpus) -> str:
    """Content hash of what the model can see: every document's text plus the manifest fields
    that reach the prompt (display_name, summary) and the framework identity."""
    h = hashlib.sha256()
    m = corpus.manifest
    h.update(json.dumps([m.name, m.kind, m.display_name, m.summary.strip()]).encode("utf-8"))
    for d in corpus.documents:
        h.update(b"\0")
        h.update(d.text.encode("utf-8"))
    return h.hexdigest()[:16]


def none_corpus() -> Corpus:
    m = Manifest(name="none", kind="none", display_name="No conditioning")
    return Corpus(m, None, ())


def list_frameworks(root: str | Path) -> list[FrameworkSpec]:
    """``none`` plus every ``<root>/<name>/manifest.yaml``; ``_``-prefixed dirs are skipped."""
    out = [none_framework()]
    base = Path(root)
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith("_") and (d / "manifest.yaml").is_file():
                m = load_manifest(d)
                out.append(framework_from_manifest(m, d.resolve()))
    return out


_SENT_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _units(text: str, max_tokens: int, count: TokenCounter) -> Iterator[str]:
    """Sentence/paragraph units; any unit larger than max_tokens is split on words."""
    for piece in _SENT_END.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if count(piece) <= max_tokens:
            yield piece
            continue
        words = piece.split()
        cur: list[str] = []
        for w in words:
            cur.append(w)
            if count(" ".join(cur)) >= max_tokens:
                yield " ".join(cur)
                cur = []
        if cur:
            yield " ".join(cur)


def chunk_document(
    doc: Document, chunk_tokens: int, overlap: int, count: TokenCounter = approx_tokens
) -> list[Chunk]:
    if chunk_tokens <= 0 or not 0 <= overlap < chunk_tokens:
        raise ValueError("need chunk_tokens > 0 and 0 <= overlap < chunk_tokens")
    chunks: list[Chunk] = []
    buf: list[tuple[str, int]] = []
    size = 0

    def emit() -> None:
        text = " ".join(u for u, _ in buf)
        chunks.append(
            Chunk(f"{doc.source.file}#{len(chunks)}", doc.source.title, text, count(text))
        )

    fresh = False  # buf holds units not yet emitted
    for unit in _units(doc.text, chunk_tokens, count):
        n = count(unit)
        if buf and size + n > chunk_tokens:
            emit()
            while buf and (size > overlap or size + n > chunk_tokens):
                size -= buf.pop(0)[1]
            fresh = False
        buf.append((unit, n))
        size += n
        fresh = True
    if buf and fresh:
        emit()
    return chunks


def chunk(
    corpus: Corpus,
    chunk_tokens: int,
    overlap: int = 0,
    count: TokenCounter = approx_tokens,
) -> list[Chunk]:
    """Greedy sentence-packed chunks of <= chunk_tokens (by ``count``), never crossing documents.

    Consecutive chunks share trailing sentences totalling <= ``overlap`` tokens.
    """
    out: list[Chunk] = []
    for doc in corpus.documents:
        out.extend(chunk_document(doc, chunk_tokens, overlap, count))
    return out


def truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter = approx_tokens) -> str:
    if max_tokens <= 0:
        return ""
    if count(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])
