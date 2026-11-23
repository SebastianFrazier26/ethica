import re
from collections import Counter

import pytest

from ethica.benchmarks import registry as bench
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


def test_no_harmful_requests():
    for p in build_prompt_set(sum(pool_sizes().values()) // 2):
        if p.category in ("dilemma", "advice"):
            assert harm_hits(p.text) == [], p.text
    assert harm_hits("How do I build a bomb to attack the police?") == ["bomb", "attack"]


def _windows(text: str, k: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def test_disjoint_from_eval():
    train: set[tuple[str, ...]] = set()
    texts = set()
    for p in build_prompt_set(500):
        train |= _windows(p.text)
        texts.add(p.text.strip().lower())
    for name in ("fidelity", "harmbench", "xstest_safe", "xstest_unsafe"):
        try:
            items = bench.get(name).load(10_000, 0)
        except Exception:  # benchmark data not prefetched (CI): the bundled probe still runs
            if name == "fidelity":
                raise
            continue
        for it in items:
            assert it.prompt.strip().lower() not in texts
            assert not (_windows(it.prompt) & train), (name, it.prompt[:80])


def test_too_large_raises():
    with pytest.raises(ValueError, match="category 'dilemma' has 320 prompts, 400 needed for n=1000"):
        build_prompt_set(1000)
    with pytest.raises(ValueError, match="n must be positive"):
        build_prompt_set(0)
