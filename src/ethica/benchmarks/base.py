"""Static-benchmark contracts and shared helpers (sampling, MC formatting, HF fetching)."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

NAN = float("nan")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class BenchItem:
    id: str
    benchmark: str
    prompt: str
    choices: list[str] | None = None
    # MC items store the correct letter ("B"); numeric items the number as str; binary "yes"/"no"
    answer: str | int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ScoreTuple(NamedTuple):
    """Maps 1:1 onto types.Score once the caller supplies episode_id."""

    metric: str
    value: float
    scorer: str
    rationale: str = ""


def unparseable_score(flag: bool, why: str = "") -> ScoreTuple:
    """Emitted next to every benchmark's primary metric so answer-extraction failures (which
    differ by arm when one arm is wordier) are visible instead of silently selecting items."""
    return ScoreTuple("benchmark_unparseable", float(flag), "exact", why)


@runtime_checkable
class Benchmark(Protocol):
    """`load(None, seed)` returns `default_n` items. Scoring is a pure function of (item, text)."""

    default_n: int
    system_prompt: str | None

    @property
    def name(self) -> str: ...

    def load(self, n: int | None = None, seed: int = 0) -> list[BenchItem]: ...

    def score(self, item: BenchItem, response_text: str) -> list[ScoreTuple]: ...


@dataclass(frozen=True)
class Source:
    """One pinned file on the HF Hub. Revision is a commit sha so reruns see identical data."""

    repo: str
    filename: str
    revision: str
    license: str


def cache_dir() -> Path:
    override = os.environ.get("ETHICA_CACHE_DIR")
    root = Path(override) if override else Path(__file__).resolve().parents[3] / "data" / "cache"
    return root / "hf"


def fetch(src: Source) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            src.repo,
            src.filename,
            repo_type="dataset",
            revision=src.revision,
            cache_dir=cache_dir(),
        )
    )


def sample_indices(total: int, n: int | None, seed: int) -> list[int]:
    if n is None or n >= total:
        return list(range(total))
    return sorted(random.Random(f"idx:{seed}").sample(range(total), n))


def stratified_sample[T](
    items: Sequence[T], key: Callable[[T], str], n: int | None, seed: int
) -> list[T]:
    """Equal allocation per stratum, remainder spread over strata chosen by seed.

    A stratum smaller than its quota hands the surplus to the others, so the
    result has exactly min(n, len(items)) elements. Output is in original order.
    """
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(f"strat:{seed}")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[key(it)].append(i)
    names = sorted(groups)
    for g in names:
        rng.shuffle(groups[g])
    quota = dict.fromkeys(names, 0)
    remaining = n
    active = list(names)
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        bonus = set(rng.sample(active, extra)) if extra else set()
        progressed = 0
        for g in list(active):
            want = base + (1 if g in bonus else 0)
            take = min(want, len(groups[g]) - quota[g])
            quota[g] += take
            progressed += take
            if quota[g] >= len(groups[g]):
                active.remove(g)
        remaining -= progressed
        if progressed == 0:
            break
    picked = sorted(i for g in names for i in groups[g][: quota[g]])
    return [items[i] for i in picked]


def item_rng(seed: int, item_id: str) -> random.Random:
    """Per-item RNG so shuffles don't depend on which other items were sampled."""
    return random.Random(f"{seed}:{item_id}")


def format_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def resolve_n(n: int | None, default_n: int) -> int:
    return default_n if n is None else n
