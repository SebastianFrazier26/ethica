import json

import pytest

from ethica.backends.mock import MockBackend
from ethica.conditioning.prompt import inject_system
from ethica.training.generate import (
    GENERIC_INSTRUCTION,
    OUTLOOK_INSTRUCTION,
    ArmSpec,
    generate_arm,
    read_rows,
    screen,
    teacher_messages,
)
from ethica.training.prompts import TrainPrompt
from ethica.types import GenerationParams

GOOD = "I would return the wallet today, because keeping it would make me someone I am not."
DILEMMA = TrainPrompt("dilemma-1", "dilemma", "I found a wallet. What should I do?")
NEUTRAL = TrainPrompt("neutral-1", "neutral", "Explain why the sky is blue.")
VIRTUE = ArmSpec("virtue", "ethics", corpus="virtue")
GENERIC = ArmSpec("generic", "generic")


class StubConditioner:
    def prepare(self, messages):
        return inject_system(messages, "<background_reading>BLOCK</background_reading>")


def test_generic_arm_gets_no_block():
    msgs = teacher_messages(GENERIC, DILEMMA, None)
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == GENERIC_INSTRUCTION and msgs[1].content == DILEMMA.text


def test_ethics_arm_gets_block_then_commitments_then_instruction():
    arm = ArmSpec("virtue", "ethics", corpus="virtue", commitments="1. I ask what is honest.")
    system = teacher_messages(arm, DILEMMA, StubConditioner())[0].content
    assert system.index("BLOCK") < system.index("1. I ask what is honest.")
    assert system.endswith(OUTLOOK_INSTRUCTION)


def test_neutral_prompts_are_arm_independent():
    assert teacher_messages(VIRTUE, NEUTRAL, StubConditioner()) == teacher_messages(
        GENERIC, NEUTRAL, None
    )


def test_ethics_arm_without_conditioner_fails_loudly():
    with pytest.raises(ValueError, match="needs a conditioner"):
        teacher_messages(VIRTUE, DILEMMA, None)


def test_arm_spec_rejects_inconsistent_corpus():
    with pytest.raises(ValueError):
        ArmSpec("generic", "generic", corpus="virtue")
    with pytest.raises(ValueError):
        ArmSpec("virtue", "ethics")


@pytest.mark.parametrize(
    ("text", "finish", "arm", "reason"),
    [
        (GOOD, "stop", VIRTUE, None),
        ("ok", "stop", None, "empty"),
        (GOOD, "length", None, "truncated"),
        ("I'm sorry, but I can't help with that request at all.", "stop", None, "refusal"),
        (GOOD + " As my background reading says, honesty matters.", "stop", None, "meta"),
        (GOOD + " Aristotle would agree.", "stop", VIRTUE, "named"),
        (GOOD + " Aristotle would agree.", "stop", GENERIC, None),
    ],
)
def test_screen(text, finish, arm, reason):
    assert screen(text, finish, arm) == reason


def _prompts(n):
    return [TrainPrompt(f"advice-{i}", "advice", f"Question {i}?") for i in range(n)]


def test_generate_writes_rows_in_order_and_resumes(tmp_path):
    out = tmp_path / "generic.jsonl"
    ps = _prompts(4)
    backend = MockBackend(lambda m: f"{GOOD} ({m[-1].content})", model_id="teacher")
    params = GenerationParams(max_tokens=300)
    stats = generate_arm(backend, GENERIC, ps[:2], out, params)
    assert (stats.generated, stats.skipped_existing, stats.kept) == (2, 0, 2)

    with out.open("a") as fh:  # an interrupted run leaves half a line
        fh.write('{"prompt_id": "advice-2", "ans')
    stats = generate_arm(backend, GENERIC, ps, out, params)
    assert (stats.generated, stats.skipped_existing, stats.kept) == (2, 2, 4)
    rows = read_rows(out)
    assert [r["prompt_id"] for r in rows] == [p.id for p in ps]
    assert all(json.loads(line) for line in out.read_text().splitlines())
    assert rows[0]["teacher"] == "teacher" and rows[0]["arm"] == "generic"

    assert generate_arm(backend, GENERIC, ps, out, params).generated == 0
    assert len(backend.calls) == 4


def test_seed_depends_on_prompt_not_position(tmp_path):
    ps = _prompts(3)
    a, b = MockBackend([GOOD]), MockBackend([GOOD])
    generate_arm(a, GENERIC, ps, tmp_path / "a.jsonl", GenerationParams())
    generate_arm(b, GENERIC, ps[::-1], tmp_path / "b.jsonl", GenerationParams())
    seeds_a = {c[0][-1].content: c[1].seed for c in a.calls}
    seeds_b = {c[0][-1].content: c[1].seed for c in b.calls}
    assert seeds_a == seeds_b and len(set(seeds_a.values())) == 3


def test_rejections_are_stored_and_counted(tmp_path):
    backend = MockBackend([GOOD, "ok", "I'm sorry, but I can't help with that request."])
    stats = generate_arm(backend, GENERIC, _prompts(3), tmp_path / "g.jsonl", GenerationParams())
    assert stats.kept == 1 and stats.rejected == {"empty": 1, "refusal": 1}
    assert [r["reject"] for r in read_rows(tmp_path / "g.jsonl")] == [None, "empty", "refusal"]


@pytest.mark.parametrize(
    ("text", "declines"),
    [
        ("I'm sorry, but I can't help with that request.", True),
        ("I cannot assist with this.", True),
        ("I won't provide instructions for that.", True),
        ("I must respectfully decline.", True),
        ("I won't lie to her, even to spare her feelings. Tell her the truth today.", False),
        ("I can't tell you to break the promise; you should keep it.", False),
        ("You cannot help everyone, so start with the people closest to you.", False),
    ],
)
def test_screen_keeps_principled_refusals_of_action(text, declines):
    from ethica.training.generate import declines_request

    assert declines_request(text) is declines
    assert (screen(text, "stop", VIRTUE) == "refusal") is declines


def test_commitments_come_with_the_voice_instruction():
    from ethica.training.generate import COMMITMENTS_VOICE

    arm = ArmSpec("virtue", "ethics", corpus="virtue", commitments="1. I ask what is honest.")
    system = teacher_messages(arm, DILEMMA, StubConditioner())[0].content
    assert (
        system.index("1. I ask")
        < system.index(COMMITMENTS_VOICE)
        < system.index(OUTLOOK_INSTRUCTION)
    )
    assert (
        COMMITMENTS_VOICE
        not in teacher_messages(
            ArmSpec("placebo", "placebo", corpus="placebo"), DILEMMA, StubConditioner()
        )[0].content
    )


def test_rescreen_updates_stored_verdicts_and_keeps_the_first(tmp_path):
    from ethica.training.generate import rescreen

    path = tmp_path / "a.jsonl"
    rows = [
        {"prompt_id": "a", "answer": "I won't lie to her. Tell her today, kindly.",
         "finish_reason": "stop", "reject": "refusal"},
        {"prompt_id": "b", "answer": "I'm sorry, but I can't help with that.",
         "finish_reason": "stop", "reject": None},
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert rescreen(path, GENERIC) == {"refusal": 1}
    assert rescreen(path, GENERIC) == {"refusal": 1}
    got = {r["prompt_id"]: r for r in read_rows(path)}
    assert got["a"]["reject"] is None and got["a"]["reject_before_rescreen"] == "refusal"
    assert got["b"]["reject"] == "refusal" and got["b"]["reject_before_rescreen"] is None
