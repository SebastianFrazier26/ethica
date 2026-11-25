import json

import pytest

from ethica.backends.mock import MockBackend
from ethica.training.generate import ArmSpec, read_rows
from ethica.training.revise import (
    NO_DEPARTURE,
    critique_messages,
    no_departure,
    revise_arm,
)
from ethica.types import GenerationParams

COMMIT = "1. I do not lie, even to bring about a good result."
DEON = ArmSpec("deontology", "ethics", corpus="deontology", commitments=COMMIT)
PARAMS = GenerationParams(max_tokens=200, temperature=0.7, seed=0)
REVISED = "You should tell your friend the truth about the dent, today, in person."


def _row(pid, category="dilemma", answer="A white lie is fine here.", reject=None):
    return {
        "prompt_id": pid,
        "category": category,
        "prompt": f"Question {pid}?",
        "answer": answer,
        "finish_reason": "stop",
        "reject": reject,
        "arm": "deontology",
        "teacher": "mock",
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _script(messages):
    if "Critique:" in messages[-1].content:
        return REVISED
    return NO_DEPARTURE if "fine" not in messages[-1].content else "Departs from 1: it lies."


def test_revises_departures_keeps_the_rest(tmp_path):
    src, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(
        src,
        [
            _row("d-1"),
            _row("d-2", answer="Tell the truth, even though it costs you."),
            _row("n-1", category="neutral", answer="Rayleigh scattering."),
            _row("d-3", reject="refusal"),
        ],
    )
    backend = MockBackend(_script)
    stats = revise_arm(backend, DEON, src, out, PARAMS)
    rows = {r["prompt_id"]: r for r in read_rows(out)}
    assert (stats.critiqued, stats.revised, stats.unchanged, stats.passed_through) == (2, 1, 1, 2)
    assert rows["d-1"]["answer"] == REVISED
    assert rows["d-1"]["original_answer"] == "A white lie is fine here."
    assert rows["d-1"]["revision"] == "revised" and rows["d-1"]["reject"] is None
    assert rows["d-2"]["revision"] == "unchanged"
    assert rows["d-2"]["answer"] == "Tell the truth, even though it costs you."
    assert rows["n-1"]["revision"] == "passthrough" and "critique" not in rows["n-1"]
    assert rows["d-3"]["reject"] == "refusal"
    # neutral and rejected rows never reach the teacher; critiques are greedy
    assert len(backend.calls) == 3
    assert all(p.temperature == 0.0 for m, p in backend.calls if "Critique:" not in m[-1].content)


def test_screened_revision_keeps_the_original(tmp_path):
    src, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(src, [_row("d-1")])
    backend = MockBackend(["Departs from 1.", "As Kant says, you must not lie to your friend."])
    stats = revise_arm(backend, DEON, src, out, PARAMS)
    row = read_rows(out)[0]
    assert stats.revision_rejected == 1 and row["revision"] == "rejected_kept_original"
    assert row["reject"] is None and row["answer"] == "A white lie is fine here."
    assert row["rejected_revision_reason"] == "named"


def test_resume_skips_done_rows_and_matches_uninterrupted(tmp_path):
    src = tmp_path / "a.jsonl"
    _write(src, [_row(f"d-{i}") for i in range(4)])
    full, part = tmp_path / "full.jsonl", tmp_path / "part.jsonl"
    revise_arm(MockBackend(_script), DEON, src, full, PARAMS)
    _write(part, read_rows(full)[:2])
    with part.open("a") as fh:
        fh.write('{"prompt_id": "d-2", "torn')
    backend = MockBackend(_script)
    stats = revise_arm(backend, DEON, src, part, PARAMS)
    assert stats.skipped_existing == 2 and len(backend.calls) == 4
    assert read_rows(part) == read_rows(full)


def test_seeds_fixed_per_prompt(tmp_path):
    src = tmp_path / "a.jsonl"
    _write(src, [_row("d-1")])
    a, b = MockBackend(_script), MockBackend(_script)
    revise_arm(a, DEON, src, tmp_path / "x.jsonl", PARAMS)
    revise_arm(b, DEON, src, tmp_path / "y.jsonl", PARAMS)
    seeds = [p.seed for _, p in a.calls]
    assert seeds == [p.seed for _, p in b.calls] and seeds[0] != seeds[1]


@pytest.mark.parametrize(
    "arm",
    [
        ArmSpec("generic", "generic"),
        ArmSpec("placebo", "placebo", corpus="placebo"),
        ArmSpec("virtue", "ethics", corpus="virtue"),
    ],
)
def test_refuses_arms_without_commitments(tmp_path, arm):
    with pytest.raises(ValueError, match="signed-off commitments"):
        revise_arm(MockBackend(["-"]), arm, tmp_path / "a", tmp_path / "b", PARAMS)


@pytest.mark.parametrize(
    ("text", "expected"), [("NONE", True), (" none. ", True), ("None of note, but 1.", False)]
)
def test_no_departure(text, expected):
    assert no_departure(text) is expected


def test_critique_shows_commitments_question_and_answer():
    user = critique_messages(COMMIT, "Q?", "A.")[1].content
    assert COMMIT in user and "Q?" in user and "A." in user
