"""Lexical-confound measures for RAG contexts (D-58).

Tests the reading that non-consequentialist corpora lower harmful compliance only because BM25
puts more harm/safety vocabulary next to harmful prompts. Densities are shares of content
tokens as `bm25.tokenise` produces them (lowercased, BM25 stopwords removed), so they count
the same tokens retrieval matched on.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ethica.conditioning.bm25 import tokenise

FloatArray = npt.NDArray[np.float64]

# Prefix-matched. Mixes descriptive harm terms (poison, weapon) with safety/normative ones
# (wrong, unsafe, consent): the confound under test is "words about harm near the request",
# whichever register they come from.
HARM_STEMS: tuple[str, ...] = (
    "harm", "injur", "kill", "murder", "violen", "weapon", "firearm", "danger", "hazard",
    "illegal", "illicit", "unlaw", "crime", "crimin", "abus", "assault", "attack", "poison",
    "toxic", "explos", "bomb", "lethal", "fatal", "wound", "threat", "terror", "fraud", "scam",
    "steal", "theft", "thie", "robber", "drug", "overdos", "unsafe", "risk", "victim", "hack",
    "malware", "ransom", "exploit", "tortur", "suicid", "traffick", "smuggl", "launder",
    "extort", "harass", "damag", "destroy", "destruct", "safety", "protect", "prohibit",
    "forbid", "punish", "prison", "arrest", "police", "offen", "cruel", "suffer", "consent",
    "wrong",
)  # fmt: skip
# Short forms whose prefixes would swallow unrelated words (pain -> paint, rob -> robot).
HARM_WORDS: frozenset[str] = frozenset((
    "gun", "guns", "die", "dies", "died", "dying", "dead", "death", "deaths", "pain", "pains",
    "painful", "hurt", "hurts", "rob", "robs", "robbed", "safe", "safer", "safely",
))  # fmt: skip
HARM_EXCLUDE_PREFIXES: tuple[str, ...] = ("harmon", "offend", "offer")

# Instruction words that every request shares; without them "overlap with the request" would
# mostly measure how often a corpus says "write" or "how".
PROMPT_FUNCTION_WORDS: frozenset[str] = frozenset((
    "about", "all", "any", "can", "could", "detailed", "did", "do", "does", "explain", "give",
    "how", "list", "make", "me", "my", "no", "not", "so", "some", "step", "steps", "tell",
    "than", "then", "there", "these", "those", "what", "when", "where", "which", "who", "why",
    "would", "write", "describe", "provide",
))  # fmt: skip


def is_harm_token(tok: str) -> bool:
    if tok in HARM_WORDS:
        return True
    return tok.startswith(HARM_STEMS) and not tok.startswith(HARM_EXCLUDE_PREFIXES)


def prompt_terms(prompt: str) -> frozenset[str]:
    return frozenset(t for t in tokenise(prompt) if t not in PROMPT_FUNCTION_WORDS)


@dataclass(frozen=True)
class Density:
    n_tokens: int
    harm_hits: int
    overlap_hits: int

    @property
    def harm(self) -> float:
        return self.harm_hits / self.n_tokens if self.n_tokens else math.nan

    @property
    def overlap(self) -> float:
        return self.overlap_hits / self.n_tokens if self.n_tokens else math.nan


def density(texts: Iterable[str], prompt: str = "") -> Density:
    """Harm-lexicon and request-overlap shares over the concatenated texts."""
    terms = prompt_terms(prompt)
    toks = [t for text in texts for t in tokenise(text)]
    return Density(
        n_tokens=len(toks),
        harm_hits=sum(1 for t in toks if is_harm_token(t)),
        overlap_hits=sum(1 for t in toks if t in terms),
    )


# ---------- correlation ----------


def rankdata(x: Sequence[float] | FloatArray) -> FloatArray:
    """Ranks starting at 1; ties get the mean of the ranks they span."""
    v = np.asarray(x, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: FloatArray, b: FloatArray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else math.nan


@dataclass(frozen=True)
class Spearman:
    rho: float
    p: float  # two-sided permutation p
    n: int
    exact: bool


_EXACT_MAX_N = 8  # 8! = 40320 orderings


def spearman(
    x: Sequence[float] | FloatArray,
    y: Sequence[float] | FloatArray,
    n_perm: int = 10000,
    seed: int = 0,
) -> Spearman:
    """Spearman rho with a permutation p-value (exact for n <= 8, else seeded Monte Carlo with
    the +1 correction). NaN pairs are dropped."""
    xv, yv = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(xv) != len(yv):
        raise ValueError("x and y must have equal length")
    ok = ~(np.isnan(xv) | np.isnan(yv))
    rx, ry = rankdata(xv[ok]), rankdata(yv[ok])
    n = len(rx)
    if n < 3:
        return Spearman(math.nan, math.nan, n, False)
    rho = _pearson(rx, ry)
    if math.isnan(rho):
        return Spearman(rho, math.nan, n, False)
    thresh = abs(rho) - 1e-12
    if n <= _EXACT_MAX_N:
        hits = total = 0
        for perm in itertools.permutations(range(n)):
            total += 1
            hits += abs(_pearson(rx, ry[list(perm)])) >= thresh
        return Spearman(rho, hits / total, n, True)
    rng = np.random.default_rng(seed)
    hits = sum(abs(_pearson(rx, rng.permutation(ry))) >= thresh for _ in range(n_perm))
    return Spearman(rho, (1 + hits) / (1 + n_perm), n, False)


# ---------- logistic regression ----------


@dataclass(frozen=True)
class Logit:
    coef: FloatArray
    se: FloatArray
    n: int
    converged: bool


def logistic_fit(
    x: FloatArray, y: Sequence[float] | FloatArray, ridge: float = 1e-6, max_iter: int = 100
) -> Logit:
    """Maximum-likelihood logistic regression by Newton-Raphson (IRLS). ``x`` must already
    contain an intercept column if one is wanted. The tiny ridge keeps the Hessian invertible
    under (quasi-)separation; SEs are the naive inverse-information ones, not clustered."""
    xm = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xm.ndim != 2 or len(xm) != len(yv):
        raise ValueError("x must be 2-D with one row per y")
    beta = np.zeros(xm.shape[1])
    pen = ridge * np.eye(xm.shape[1])
    converged = False
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(xm @ beta)))
        w = p * (1.0 - p)
        grad = xm.T @ (yv - p) - pen @ beta
        hess = (xm * w[:, None]).T @ xm + pen
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if float(np.abs(step).max()) < 1e-8:
            converged = True
            break
    p = 1.0 / (1.0 + np.exp(-(xm @ beta)))
    hess = (xm * (p * (1.0 - p))[:, None]).T @ xm + pen
    se = np.sqrt(np.diag(np.linalg.inv(hess)))
    return Logit(beta, se, len(yv), converged)
