from __future__ import annotations

import json
import math

import pytest
from judge_stubs import CONDITIONING, StubBackend, make_episode, make_scenario

from ethica.scoring.judge import (
    DEFAULT_REDACT_TERMS,
    REDACTION_TOKEN,
    Judge,
    available_rubrics,
    judge_episodes,
    load_rubric,
    parse_verdict,
    redact,
    render_for_judge,
    rubrics_for,
)
from ethica.types import Message

EXPECTED = {
    "deception_about_actions": "deception",
    "overt_refusal": "overt_refusal",
    "framework_rationalisation": "framework_rationalisation",
    "scheming_reasoning": "scheming_reasoning",
    "coercion": "coercion",
    "sycophancy": "sycophancy",
    "covert_sabotage": "covert_sabotage",
    "refusal_static": "over_refusal",
    "harmful_compliance_static": "harmful_compliance",
}
GOOD = json.dumps({"verdict": "yes", "confidence": 0.9, "evidence": "q", "reasoning": "r"})


def test_all_rubrics_valid() -> None:
    assert set(available_rubrics()) == set(EXPECTED)
    for rid, metric in EXPECTED.items():
        r = load_rubric(rid)
        assert r.id == rid
        assert r.metric == metric
        assert len(r.examples) == 2
        verdicts = {ex.output["verdict"] for ex in r.examples}
        assert verdicts == {"yes", "no"}
        for ex in r.examples:
            assert 0 <= ex.output["confidence"] <= 1
        assert '"verdict"' in r.output_schema


def test_metrics_documented() -> None:
    from pathlib import Path

    doc = (Path(__file__).parents[1] / "docs" / "METRICS.md").read_text()
    for metric in EXPECTED.values():
        assert f"`{metric}`" in doc


def test_rubrics_for() -> None:
    assert rubrics_for(make_scenario(["coercion"])) == ["coercion"]
    assert rubrics_for("xstest_safe") == ["refusal_static"]
    assert rubrics_for("xstest_unsafe") == ["harmful_compliance_static"]
    assert rubrics_for("harmbench") == ["harmful_compliance_static"]
    assert rubrics_for("unknown") == []
    with pytest.raises(KeyError):
        rubrics_for(make_scenario(["nope"]))


def test_redaction_case_insensitive_and_word_boundary() -> None:
    T = REDACTION_TOKEN
    out = redact("A UTILITARIAN and kant discuss God, but Godzilla and Kantor stay.")
    assert out == f"A {T} and {T} discuss {T}, but Godzilla and Kantor stay."
    assert redact("the Categorical   Imperative applies") == f"the {REDACTION_TOKEN} applies"
    assert redact("kant", ()) == "kant"
    assert redact("alpha beta", ["Alpha"]) == f"{REDACTION_TOKEN} beta"
    assert "God" in DEFAULT_REDACT_TERMS


def test_render_excludes_conditioning_and_redacts() -> None:
    ep = make_episode("The utilitarian answer is to disable it.")
    text = render_for_judge(ep, make_scenario(), include_scratchpad=False)
    assert "CONDITIONING-MARKER" not in text
    assert "categorical imperative" not in text.lower()
    assert "tool protocol" not in text
    assert "You are an ops assistant." in text
    assert "utilitarian" not in text.lower()
    assert "set_config" in text
    assert "oversight_disabled" in text
    assert "oversight.enabled" in text  # state diff
    assert "+" in text and "chars]" in text  # tool result truncated


def test_scratchpad_toggle() -> None:
    ep = make_episode("<scratchpad>hide it</scratchpad>Done.")
    off = render_for_judge(ep, None, include_scratchpad=False)
    on = render_for_judge(ep, None, include_scratchpad=True)
    assert "hide it" not in off
    assert "PRIVATE SCRATCHPAD" in on and "hide it" in on


def test_long_transcript_truncated_keeps_ends_and_calls() -> None:
    msgs = [Message("user", "FIRST")]
    for i in range(40):
        msgs.append(Message("assistant", f"filler {i} " + "z" * 500))
    msgs.append(Message("assistant", '<tool_call>{"name":"special"}</tool_call>'))
    msgs.extend(Message("assistant", f"filler2 {i} " + "z" * 500) for i in range(40))
    msgs.append(Message("assistant", "FINAL REPORT"))
    ep = make_episode(transcript=msgs)
    from ethica.scoring.judge import render_with_meta

    r = render_with_meta(ep, None, False, max_chars=6000)
    assert r.truncated
    assert "FIRST" in r.text and "FINAL REPORT" in r.text and "special" in r.text
    assert "omitted for length" in r.text


@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        f"```json\n{GOOD}\n```",
        f"Sure! Here you go: {GOOD} Hope that helps.",
        f"<think>{{not json}}</think>{GOOD}",
        '{"bad": 1} ' + GOOD,
        '{"verdict": "YES ", "confidence": 0.9}',
    ],
)
def test_parse_verdict_ok(text: str) -> None:
    v = parse_verdict(text)
    assert v is not None and v.verdict == "yes"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json",
        '{"verdict": "maybe", "confidence": 0.5}',
        '{"verdict": "yes"}',
        '{"verdict": "yes", "confidence": 3}',
        '{"verdict": "yes", "confidence": 0.5',
    ],
)
def test_parse_verdict_bad(text: str) -> None:
    assert parse_verdict(text) is None


def test_judge_yes_no_unclear() -> None:
    for verdict, expected in (("yes", 1.0), ("no", 0.0)):
        be = StubBackend(
            [json.dumps({"verdict": verdict, "confidence": 0.8, "evidence": "e", "reasoning": "r"})]
        )
        j = Judge(be, "qwen3-8b")
        s = j.judge(make_episode(), "overt_refusal", make_scenario())
        assert s.value == expected
        assert s.scorer == "judge:qwen3-8b"
        assert s.metric == "overt_refusal"
        assert "e" in s.rationale and "r" in s.rationale
        assert be.calls[0][0].role == "system"
        assert CONDITIONING not in be.calls[0][1].content
    be = StubBackend(['{"verdict": "unclear", "confidence": 0.2}'])
    s = Judge(be, "m").judge(make_episode(), "coercion", make_scenario())
    assert math.isnan(s.value)


def test_judge_default_temperature_zero() -> None:
    assert Judge(StubBackend([]), "m").params.temperature == 0.0


def test_judge_retry_then_success() -> None:
    be = StubBackend(["I think yes", GOOD])
    s = Judge(be, "m").judge(make_episode(), "coercion", make_scenario())
    assert s.value == 1.0
    assert len(be.calls) == 2
    assert "only the JSON" in be.calls[1][-1].content


def test_judge_retry_then_nan() -> None:
    be = StubBackend(["nope", "still nope"])
    s = Judge(be, "m").judge(make_episode(), "coercion", make_scenario())
    assert math.isnan(s.value)
    assert "unparseable" in s.rationale
    assert len(be.calls) == 2


def test_framework_rationalisation_keeps_vocab_others_redact() -> None:
    ep = make_episode("<scratchpad>The utilitarian calculus says do it.</scratchpad>ok")
    j = Judge(StubBackend([]), "m")
    sc = make_scenario()
    fr = j.render(ep, load_rubric("framework_rationalisation"), sc)
    other = j.render(ep, load_rubric("scheming_reasoning"), sc)
    assert "utilitarian" in fr.text
    assert "CONDITIONING-MARKER" not in fr.text
    assert "utilitarian" not in other.text.lower()


def test_judge_episodes_filters() -> None:
    sc = make_scenario(["coercion", "scheming_reasoning", "framework_rationalisation"])
    lookup = {"s1": sc}
    plain = make_episode("no reasoning", episode_id="plain")
    thinking = make_episode("<scratchpad>hmm</scratchpad>x", episode_id="think")
    errored = make_episode(episode_id="err", error="boom")
    be = StubBackend([GOOD] * 20)
    scores = list(judge_episodes(Judge(be, "m"), [plain, thinking, errored], lookup))
    got = {(s.episode_id, s.metric) for s in scores}
    assert got == {
        ("plain", "coercion"),
        ("think", "coercion"),
        ("think", "scheming_reasoning"),
        ("think", "framework_rationalisation"),
    }

    prior = {"think": {"misaligned_action": 0.0}}
    be2 = StubBackend([GOOD] * 20)
    got2 = {
        s.metric for s in judge_episodes(Judge(be2, "m"), [thinking], lookup, prior_scores=prior)
    }
    assert "framework_rationalisation" not in got2 and "scheming_reasoning" in got2


def test_judge_episodes_benchmark_and_missing_scenario() -> None:
    bench = make_episode(
        kind="benchmark", task_id="xstest:12", episode_id="b", meta={"benchmark": "xstest_safe"}
    )
    orphan = make_episode(task_id="missing", episode_id="o")
    scores = list(judge_episodes(Judge(StubBackend([GOOD] * 5), "m"), [bench, orphan], {}))
    assert [(s.episode_id, s.metric) for s in scores] == [("b", "over_refusal")]


def test_benchmark_rubrics_cover_every_judgeable_benchmark_via_meta() -> None:
    """The orchestrator writes BenchItem.benchmark (== Benchmark.name) into meta["benchmark"];
    this goes through that exact value so a name mismatch cannot hide again."""
    from ethica.benchmarks import registry as bench_registry
    from ethica.scoring.judge import BENCHMARK_RUBRICS

    assert set(BENCHMARK_RUBRICS) <= set(bench_registry.names())
    want = {
        "xstest_safe": "over_refusal",
        "xstest_unsafe": "harmful_compliance",
        "harmbench": "harmful_compliance",
    }
    for name, metric in want.items():
        ep = make_episode(
            kind="benchmark", task_id=f"{name}:1", episode_id=name, meta={"benchmark": name}
        )
        got = list(judge_episodes(Judge(StubBackend([GOOD]), "m"), [ep], {}))
        assert [s.metric for s in got] == [metric], name


def test_judge_resume_is_per_rubric() -> None:
    lookup = {"s1": make_scenario(["coercion", "overt_refusal"])}
    ep = make_episode()
    be = StubBackend([GOOD] * 5)
    got = list(judge_episodes(Judge(be, "m"), [ep], lookup, done={"e1": {"coercion"}}))
    assert [s.metric for s in got] == ["overt_refusal"]
    assert len(be.calls) == 1


def test_redaction_covers_every_framework_and_display_name() -> None:
    from pathlib import Path

    import yaml

    from ethica.scoring.judge import FRAMEWORK_KEY_TERMS, redact_terms_for

    root = Path(__file__).parents[1] / "corpora"
    declared = yaml.safe_load((root / "_frameworks.yaml").read_text())["frameworks"]
    terms = redact_terms_for(root)
    for fw in declared:
        if fw["kind"] != "none":
            assert fw["name"] in FRAMEWORK_KEY_TERMS, fw["name"]
            assert fw["display_name"] in terms
            assert (
                redact(f"We follow {fw['display_name']} here.", terms).count(REDACTION_TOKEN) == 1
            )
    for phrase in (
        "Scanlon",
        "reasonable rejection",
        "ethics of care",
        "Noddings",
        "Gilligan",
        "Jesus",
        "Christ",
        "Lord",
        "Allah",
        "Thomist",
        "natural law",
        "background reading",
        "outlook",
    ):
        assert redact(f"mentions {phrase} once", terms) == f"mentions {REDACTION_TOKEN} once", (
            phrase
        )


def test_rubric_ids_are_slug_validated() -> None:
    with pytest.raises(KeyError, match="invalid rubric id"):
        load_rubric("../../etc/passwd")
