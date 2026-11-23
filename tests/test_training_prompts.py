import hashlib
import re
from collections import Counter

import pytest

from ethica.benchmarks import registry as bench
from ethica.training import prompts
from ethica.training.prompts import SHARES, build_prompt_set, harm_hits, pool_sizes


def test_shares_and_size():
    ps = build_prompt_set(500)
    assert len(ps) == 500
    assert len({p.text for p in ps}) == len({p.id for p in ps}) == 500
    assert Counter(p.category for p in ps) == {
        "dilemma": 200,
        "advice": 100,
        "self": 75,
        "neutral": 125,
    }
    assert sum(SHARES.values()) == pytest.approx(1.0)


def test_deterministic_seeded_and_nested():
    assert build_prompt_set(500) == build_prompt_set(500)
    assert build_prompt_set(500, seed=1) != build_prompt_set(500)
    big = {p.id for p in build_prompt_set(500)}
    assert {p.id for p in build_prompt_set(100)} <= big


FULL = 2000


def test_full_study_size_is_supported():
    ps = build_prompt_set(FULL)
    assert len(ps) == len({p.text for p in ps}) == len({p.id for p in ps}) == FULL
    counts = Counter(p.category for p in ps)
    assert counts == {"dilemma": 800, "advice": 400, "self": 300, "neutral": 500}
    sizes = pool_sizes()
    assert all(sizes[c] >= n for c, n in counts.items())


def test_pilot_set_is_pinned():
    # the generic arm's teacher answers are keyed by these ids; growing the pool must not move them
    for n, digest in ((100, "13dabd43335eda25"), (500, "7afc1362772deff3")):
        ids = "\n".join(p.id for p in build_prompt_set(n))
        assert hashlib.sha256(ids.encode()).hexdigest()[:16] == digest
    full = {p.id for p in build_prompt_set(FULL)}
    assert {p.id for p in build_prompt_set(500)} <= full


def test_no_harmful_requests():
    # the whole pool, not only what a 2,000-prompt set happens to select
    for category in ("dilemma", "advice"):
        for text in prompts._ordered(category, 0):
            assert harm_hits(text) == [], text
    assert harm_hits("How do I build a bomb to attack the police?") == ["bomb", "attack"]


def _windows(text: str, k: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def test_disjoint_from_eval():
    train: set[tuple[str, ...]] = set()
    texts = set()
    for p in build_prompt_set(FULL):
        train |= _windows(p.text)
        texts.add(p.text.strip().lower())
    for name in ("fidelity", "open_dilemmas", "harmbench", "xstest_safe", "xstest_unsafe"):
        try:
            items = bench.get(name).load(10_000, 0)
        except Exception:  # benchmark data not prefetched (CI): the bundled probe still runs
            if name in ("fidelity", "open_dilemmas"):
                raise
            continue
        for it in items:
            assert it.prompt.strip().lower() not in texts
            assert not (_windows(it.prompt) & train), (name, it.prompt[:80])


def test_too_large_raises():
    with pytest.raises(ValueError, match=r"category '\w+' has \d+ prompts, \d+ needed for n=5000"):
        build_prompt_set(5000)
    with pytest.raises(ValueError, match="n must be positive"):
        build_prompt_set(0)
