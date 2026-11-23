import dataclasses
import re

import pytest

from ethica.agent_loop import run_episode
from ethica.backends.mock import MockBackend
from ethica.conditioning.prompt import NoneConditioner
from ethica.scenarios.registry import list_scenarios
from ethica.training.toolformat import (
    FAMILIES,
    SUCCESS_TAG,
    Step,
    ToolTaskError,
    build_tool_tasks,
    render_transcript,
    tool_rows,
)

N = 4 * len(FAMILIES)


def test_every_family_builds_a_valid_transcript():
    tasks = build_tool_tasks(N)
    assert {t.scenario.title for t in tasks} == {f.__name__.lstrip("_") for f in FAMILIES}
    for task in tasks:
        msgs = render_transcript(task)
        assert [m["role"] for m in msgs[:2]] == ["system", "user"]
        assert "## Tools" in msgs[0]["content"] and "<tool_call>" in msgs[0]["content"]
        assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"].endswith("<done/>")
        assert all('ok="true"' in m["content"] for m in msgs[3::2] if m["role"] == "user")


def test_transcript_is_what_the_harness_records():
    # replay the gold assistant turns through the real agent loop: same messages, clean finish
    for task in build_tool_tasks(len(FAMILIES)):
        msgs = render_transcript(task)
        gold = [m["content"] for m in msgs if m["role"] == "assistant"]
        ep = run_episode(
            MockBackend(gold, on_exhausted="raise"), NoneConditioner(), task.scenario, "v", 0
        )
        assert ep.error is None and ep.meta["stop_reason"] == "done"
        assert ep.meta["parse_failures"] == 0 and ep.meta["protocol_violations"] == 0
        assert [(m.role, m.content) for m in ep.transcript] == [
            (m["role"], m["content"]) for m in msgs
        ]
        assert SUCCESS_TAG in {t for ev in ep.events for t in ev.tags}
        assert all(ev.ok for ev in ep.events)


def test_deterministic_and_seeded():
    assert tool_rows(N) == tool_rows(N)
    assert tool_rows(N, seed=1) != tool_rows(N)


def test_one_row_per_assistant_turn():
    tasks = build_tool_tasks(N)
    rows = tool_rows(N)
    assert len(rows) == sum(len(t.steps) + 1 for t in tasks)
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["messages"][-1]["role"] == "assistant"
        assert r["messages"][0]["role"] == "system"
    assert any("<tool_call>" in r["messages"][-1]["content"] for r in rows)
    assert any(r["messages"][-1]["content"].endswith("<done/>") for r in rows)


def _windows(text, k=8):
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def test_disjoint_from_evaluation_scenarios():
    eval_scenarios = list_scenarios()
    assert eval_scenarios
    eval_windows = set()
    for s in eval_scenarios:
        eval_windows |= _windows(s.system_prompt) | _windows(" ".join(s.user_turns))
    eval_files = {p for s in eval_scenarios for p in s.world.files}
    for task in build_tool_tasks(N):
        sc = task.scenario
        assert sc.id not in {s.id for s in eval_scenarios}
        assert not (_windows(sc.system_prompt) | _windows(" ".join(sc.user_turns))) & eval_windows
        assert not set(sc.world.files) & eval_files
        # chores only: nothing that an evaluation scenario treats as a temptation
        assert not {"upload", "delete_file", "submit"} & set(sc.world.tools)
        assert "oversight" not in str(sc.world.config).lower()


def test_broken_task_cannot_be_built():
    task = build_tool_tasks(1)[0]
    bad_call = dataclasses.replace(
        task, steps=(Step("x", (("read_file", {"path": "/nope.txt"}),)), *task.steps[1:])
    )
    with pytest.raises(ToolTaskError, match="read_file failed"):
        render_transcript(bad_call)
    unfinished = dataclasses.replace(task, steps=task.steps[:1])
    with pytest.raises(ToolTaskError, match="success condition never fired"):
        render_transcript(unfinished)
