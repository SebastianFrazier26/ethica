"""Per-turn BM25 retrieval conditioning."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ethica.conditioning.bm25 import BM25Index
from ethica.conditioning.corpus import (
    Corpus,
    TokenCounter,
    approx_tokens,
    chunk,
    truncate_to_tokens,
)
from ethica.conditioning.prompt import inject_system, render_block
from ethica.types import FrameworkSpec, Message

INDEX_CHUNK_TOKENS = 200
INDEX_OVERLAP = 40
_TAGS = re.compile(r"<[^>]+>")
MAX_ASSISTANT_QUERY_CHARS = 1000


def build_index(
    corpus: Corpus,
    chunk_tokens: int = INDEX_CHUNK_TOKENS,
    overlap: int = INDEX_OVERLAP,
    count: TokenCounter = approx_tokens,
) -> BM25Index:
    chunks = chunk(corpus, chunk_tokens, overlap, count)
    return BM25Index([c.chunk_id for c in chunks], [c.text for c in chunks])


def build_and_save_index(corpus: Corpus, path: str | Path) -> BM25Index:
    idx = build_index(corpus)
    idx.save(path)
    return idx


def build_query(messages: list[Message]) -> str:
    """Latest user turn (tool results arrive as user turns, D-07) + last assistant turn."""
    user = next((m.content for m in reversed(messages) if m.role in ("user", "tool")), "")
    asst = next((m.content for m in reversed(messages) if m.role == "assistant"), "")
    return _TAGS.sub(" ", user) + "\n" + _TAGS.sub(" ", asst[-MAX_ASSISTANT_QUERY_CHARS:])


_GOLDEN = 0.6180339887498949


class RAGConditioner:
    """Same FRAMING and total block budget as PromptConditioner; excerpts are retrieved per turn.

    The block is padded to the budget with next-best chunks, and with deterministic evenly
    spread chunks when BM25 has too few positive hits. Otherwise a query with no lexical
    overlap would get a near-empty block and block size would vary with the condition.
    """

    def __init__(
        self,
        framework: FrameworkSpec,
        index: BM25Index,
        k: int,
        budget_tokens: int,
        count: TokenCounter = approx_tokens,
    ) -> None:
        self.framework = framework
        self.index = index
        self.k = k
        self.budget_tokens = budget_tokens
        self.count = count
        self.last_hits: list[str] = []
        self.last_block_tokens = 0
        self.turn_block_tokens: list[int] = []
        self._header_tokens = count(render_block(framework, [], budget_tokens, count))

    def _ranked(self, messages: list[Message]) -> list[int]:
        s = self.index.scores(build_query(messages))
        order = np.lexsort((np.arange(len(s)), -s))
        ranked = [int(i) for i in order if s[i] > 0.0]
        seen = set(ranked)
        # golden-ratio sequence: low-discrepancy, so fallback chunks spread over the corpus
        n = len(self.index)
        j = 0
        while len(seen) < n:
            i = int(((j + 1) * _GOLDEN % 1.0) * n)
            j += 1
            if i not in seen:
                seen.add(i)
                ranked.append(i)
            elif j > 20 * n:
                ranked.extend(i for i in range(n) if i not in seen)
                break
        return ranked

    def retrieve(self, messages: list[Message]) -> list[str]:
        cap = max(1, self.budget_tokens // max(1, self.k))
        used = self._header_tokens
        ids: list[str] = []
        texts: list[str] = []
        for i in self._ranked(messages):
            if used >= self.budget_tokens:
                break
            text = truncate_to_tokens(self.index.texts[i], cap, self.count)
            if not text:
                continue
            ids.append(self.index.ids[i])
            texts.append(text)
            used += self.count(text) + 1
        self.last_hits = ids
        return texts

    def prepare(self, messages: list[Message]) -> list[Message]:
        block = render_block(
            self.framework, self.retrieve(messages), self.budget_tokens, self.count
        )
        self.last_block_tokens = self.count(block)
        self.turn_block_tokens.append(self.last_block_tokens)
        return inject_system(messages, block)
