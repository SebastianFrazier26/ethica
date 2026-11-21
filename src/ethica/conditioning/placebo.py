"""Build the placebo corpus: factual, non-normative encyclopaedic text (D-04).

Source: Simple English Wikipedia via the HF dataset ``wikimedia/wikipedia`` (config
``20231101.simple``; licenses cc-by-sa-3.0 + gfdl, verified on the Hub 2026-11-16). Simple
English keeps articles short and plain-register, which suits a length-matched control.

HF dumps carry no category metadata, so "geography / astronomy / geology / weather" is enforced
with a keyword filter instead: an article must be dense in TOPIC_STEMS and nearly free of
NORMATIVE_STEMS (moral, religious, political, violence vocabulary). Both lists are heuristics,
not a guarantee of neutrality; the manifest records the filter so the choice is auditable.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from ethica.conditioning.corpus import Manifest, Source, approx_tokens

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.simple"
# Dataset commit sha looked up via HfApi on 2026-11-16 (last modified 2024-01-09); pinned so a
# rebuilt placebo corpus draws from the same dump.
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
LICENSE = "CC BY-SA 3.0 / GFDL"

TOPIC_STEMS = (
    "river", "mountain", "volcan", "earthquake", "glacier", "island", "ocean", "lake", "climate",
    "rainfall", "atmospher", "weather", "desert", "plate", "mineral", "sediment", "fossil",
    "geolog", "erosion", "planet", "orbit", "solar", "telescope", "comet", "asteroid",
    "latitude", "longitude", "kilometre", "elevation", "valley", "coast", "peninsula",
    "continent", "crater", "lava", "magma", "crust", "tropical", "hemisphere", "galaxy",
    "evaporat", "condens", "precipitat", "glacial", "tectonic", "seismic", "equator",
)  # fmt: skip

# Short stems collide with unrelated words by prefix ("sin"->"single", "war"->"warm"), so the
# ambiguous ones are matched as whole words instead.
NORMATIVE_STEMS = (
    "christian", "muslim", "islam", "jewish", "hindu", "buddhis", "religio", "bible", "moral",
    "ethic", "sacred", "worship", "politic", "government", "election", "president", "parliament",
    "democra", "communis", "fascis", "nazi", "dictator", "justice", "murder", "massacre",
    "genocide", "slave", "crime", "criminal", "terror", "battle", "weapon", "abortion", "racis",
    "discriminat", "church", "kill", "military", "soldier", "empire", "monarch", "legislat",
)  # fmt: skip
NORMATIVE_WORDS = frozenset(
    [
        "god",
        "gods",
        "sin",
        "sins",
        "evil",
        "holy",
        "pray",
        "soul",
        "heaven",
        "hell",
        "war",
        "wars",
        "law",
        "laws",
        "right",
        "rights",
        "vote",
        "court",
        "party",
        "army",
        "should",
        "ought",
        "wrong",
        "king",
        "queen",
    ]
)

_WORD = re.compile(r"[a-z]+")
MIN_WORDS = 200
MIN_TOPIC_PER_100 = 3.0
MIN_TOPIC_DISTINCT = 6
MAX_NORMATIVE_PER_100 = 0.25
_SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Timeline)|\(disambiguation\)")


def _rate(
    text: str, stems: tuple[str, ...], words_exact: frozenset[str] = frozenset()
) -> tuple[float, int, int]:
    """(hits per 100 words, word count, distinct matched terms)."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0, 0, 0
    hit = [w for w in words if w.startswith(stems) or w in words_exact]
    return 100.0 * len(hit) / len(words), len(words), len(set(hit))


def accept(title: str, text: str) -> bool:
    if _SKIP_TITLE.search(title):
        return False
    topic, n, distinct = _rate(text, TOPIC_STEMS)
    if n < MIN_WORDS or topic < MIN_TOPIC_PER_100 or distinct < MIN_TOPIC_DISTINCT:
        return False
    normative, _, _ = _rate(text, NORMATIVE_STEMS, NORMATIVE_WORDS)
    return normative <= MAX_NORMATIVE_PER_100


def _stream() -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True, revision=REVISION)
    yield from ds


def build_placebo_corpus(
    out_dir: str | Path,
    target_tokens: int,
    seed: int = 0,
    rows: Iterable[dict[str, Any]] | None = None,
) -> Manifest:
    """Stream articles, keep accepted ones until ``target_tokens`` (approx), write corpus.

    ``rows`` (dicts with title/text/url) lets tests inject data without network. The stream is
    consumed in dataset order; ``seed`` only shuffles the order of the kept articles on disk
    so the manifest ordering is not alphabetical.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[str, str, str]] = []
    total = 0
    seen: set[str] = set()
    for row in rows if rows is not None else _stream():
        title, text = str(row["title"]), str(row["text"]).strip()
        if title in seen or not accept(title, text):
            continue
        seen.add(title)
        kept.append((title, text, str(row.get("url") or "")))
        total += approx_tokens(text)
        if total >= target_tokens:
            break
    random.Random(seed).shuffle(kept)

    sources: list[Source] = []
    for i, (title, text, url) in enumerate(kept):
        fname = f"{i:04d}.txt"
        (out / fname).write_text(f"{title}\n\n{text}\n", encoding="utf-8")
        sources.append(
            Source(
                title=title,
                author="Wikipedia contributors",
                year=2023,
                url=url or None,
                license=LICENSE,
                file=fname,
            )
        )
    manifest = Manifest(
        name=out.name,
        kind="placebo",
        display_name="Physical geography, geology and astronomy",
        summary=("Factual encyclopaedic material on landforms, weather, geology and astronomy."),
        sources=sources,
    )
    doc = manifest.model_dump(mode="json")
    doc["provenance"] = {
        "dataset": f"{DATASET}:{CONFIG}",
        "dataset_revision": REVISION,
        "built_for_target_tokens": target_tokens,
        "approx_tokens": total,
        "seed": seed,
        "filter": {
            "min_words": MIN_WORDS,
            "min_topic_hits_per_100_words": MIN_TOPIC_PER_100,
            "min_distinct_topic_terms": MIN_TOPIC_DISTINCT,
            "max_normative_hits_per_100_words": MAX_NORMATIVE_PER_100,
            "topic_stems": list(TOPIC_STEMS),
            "normative_stems": list(NORMATIVE_STEMS),
            "normative_words": sorted(NORMATIVE_WORDS),
        },
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return manifest
