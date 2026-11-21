"""BM25 Okapi, from scratch (D-14).

Idf uses the non-negative form ln(1 + (N - df + 0.5) / (df + 0.5)) so very common terms never
contribute negative scores; this is the Lucene/Elasticsearch variant, not the original Robertson
form.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "which",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenise(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in stopwords]


@dataclass(frozen=True)
class Hit:
    index: int
    score: float


class BM25Index:
    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self.ids = list(ids)
        self.texts = list(texts)
        self.k1, self.b = k1, b
        self.stopwords = stopwords
        self._build()

    def _build(self) -> None:
        tokenised = [tokenise(t, self.stopwords) for t in self.texts]
        self.doc_len = np.array([len(t) for t in tokenised], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(tokenised) else 0.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(tokenised):
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))
        n = len(tokenised)
        self._docs: dict[str, np.ndarray] = {}
        self._tf: dict[str, np.ndarray] = {}
        self._idf: dict[str, float] = {}
        for term, plist in postings.items():
            self._docs[term] = np.array([d for d, _ in plist], dtype=np.int64)
            self._tf[term] = np.array([f for _, f in plist], dtype=np.float64)
            df = len(plist)
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def __len__(self) -> int:
        return len(self.texts)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.texts), dtype=np.float64)
        if not len(self.texts) or self.avgdl == 0.0:
            return s
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        # repeated query terms are counted once (set semantics)
        for term in set(tokenise(query, self.stopwords)):
            docs = self._docs.get(term)
            if docs is None:
                continue
            tf = self._tf[term]
            s[docs] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm[docs])
        return s

    def search(self, query: str, k: int) -> list[Hit]:
        """Top-k documents with positive score; ties broken by lower index (deterministic)."""
        s = self.scores(query)
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        return [Hit(int(i), float(s[i])) for i in order if s[i] > 0.0]

    def save(self, path: str | Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "stopwords": sorted(self.stopwords),
            "ids": self.ids,
            "texts": self.texts,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(p["ids"], p["texts"], p["k1"], p["b"], frozenset(p["stopwords"]))
