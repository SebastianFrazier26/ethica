import json
import math
import re
from pathlib import Path

from orch_helpers import Tracker, make_cfg
from rich.console import Console

from ethica import orchestrator as orch
from ethica.backends.mock import MockBackend
from ethica.benchmarks import registry as bench
from ethica.benchmarks.dilemmas import load_open_dilemmas, split_item_id
from ethica.scoring.judge import (
    NO_FRAMEWORK,
    IdentifyOption,
    Judge,
    build_identify_prompt,
    identification_order,
    identify_metric,
    identify_options_for,
    judge_episodes,
    parse_identification,
)
from ethica.tracking.store import Store
from ethica.training.prompts import build_prompt_set
from ethica.types import EpisodeResult, Message

QUIET = Console(quiet=True)
REAL_CORPORA = str(Path(__file__).parent.parent / "corpora")  # only _frameworks.yaml is read
OPTS = (
    IdentifyOption("deontology", "Duties bind regardless of outcome."),
    IdentifyOption("virtue", "What a person of good character would do."),
    IdentifyOption("act_utilitarian", "Greatest total well-being."),
)


def _ep(
    eid="e1",
    task="open_dilemmas:od01:a",
    text="As Kant says, I would keep my promise because duty binds.",
):
    return EpisodeResult(
        eid, "v", task, "benchmark", 0,
        [Message("user", "Q?"), Message("assistant", text)], [], {}, 1, 1, 1, 0.1,
        meta={"benchmark": "open_dilemmas"},
    )  # fmt: skip


def _windows(text, k=8):
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def test_benchmark_emits_both_phrasings_of_each_case():
    items = bench.get("open_dilemmas").load(None, 0)
    assert len(items) == 2 * len(load_open_dilemmas()) == 48
    cases = [split_item_id(i.id) for i in items]
    assert {p for _, p in cases} == {"a", "b"}
    assert all(cases[k][0] == cases[k + 1][0] for k in range(0, len(cases), 2))
    assert len(bench.get("open_dilemmas").load(10, 0)) == 10
    b = bench.get("open_dilemmas")
    assert b.score(items[0], "word " * 20)[0].value == 0.0
    assert b.score(items[0], "no")[0].value == 1.0


def test_dilemmas_are_held_out_from_training_and_the_probe():
    seen = set()
    for p in build_prompt_set(500):
        seen |= _windows(p.text)
    for it in bench.get("fidelity").load(None, 0):
        seen |= _windows(it.prompt)
    for d in load_open_dilemmas():
        assert not (_windows(d.a) | _windows(d.b)) & seen, d.id
        assert d.a != d.b


def test_options_come_from_framework_summaries():
    opts = identify_options_for(REAL_CORPORA)
    assert len(opts) == 7 and all(o.description and o.name not in ("none", "placebo") for o in opts)
    assert identify_options_for("/nonexistent") == ()


def test_prompt_hides_names_and_shuffles_per_episode():
    order = identification_order("e1", OPTS)
    assert order[-1] == NO_FRAMEWORK and sorted(order[:-1]) == sorted(o.name for o in OPTS)
    assert identification_order("e1", OPTS) == order
    assert len({tuple(identification_order(f"e{i}", OPTS)) for i in range(30)}) > 1
    prompt = build_identify_prompt(order, OPTS)
    for o in OPTS:
        assert o.description in prompt and o.name not in prompt
    assert "D. No single one of these" in prompt


def test_parse_identification():
    assert parse_identification('{"choice": "b", "confidence": 0.8}', "ABCD").choice == "B"
    assert (
        parse_identification('noise {"choice": "B.", "evidence": "x"} tail', "ABCD").choice == "B"
    )
    assert parse_identification('{"choice": "Z"}', "ABCD") is None
    assert parse_identification("no json", "ABCD") is None


def test_identify_scores_one_hot_and_maps_letter_back_to_framework():
    ep = _ep()
    order = identification_order(ep.episode_id, OPTS)
    letter = "ABCD"[order.index("deontology")]
    backend = MockBackend([json.dumps({"choice": letter, "confidence": 0.9, "evidence": "duty"})])
    scores = Judge(backend, "j", identify_options=OPTS).identify(ep)
    got = {s.metric: s.value for s in scores}
    assert got == {
        identify_metric("deontology"): 1.0,
        identify_metric("virtue"): 0.0,
        identify_metric("act_utilitarian"): 0.0,
        identify_metric(NO_FRAMEWORK): 0.0,
    }
    assert all(s.scorer == "judge:j" for s in scores)
    shown = backend.calls[0][0][-1].content
    assert "[REDACTED]" in shown and "Kant" not in shown and "duty binds" in shown


def test_identify_retries_then_gives_nan():
    backend = MockBackend(["garbage", "still garbage"])
    scores = Judge(backend, "j", identify_options=OPTS).identify(_ep())
    assert len(backend.calls) == 2 and len(scores) == 4
    assert all(math.isnan(s.value) for s in scores)


def test_judge_episodes_identifies_only_open_dilemmas_and_resumes():
    reply = json.dumps({"choice": "A", "confidence": 0.5})
    j = Judge(MockBackend([reply]), "j", identify_options=OPTS)
    other = _ep("e2", "fidelity:fid01")
    other.meta["benchmark"] = "fidelity"
    out = list(judge_episodes(j, [_ep(), other], {}))
    assert {s.episode_id for s in out} == {"e1"} and len(out) == 4
    done = {"e1": {identify_metric(NO_FRAMEWORK)}}
    assert list(judge_episodes(j, [_ep()], {}, done=done)) == []
    assert list(judge_episodes(Judge(MockBackend([reply]), "j"), [_ep()], {})) == []


def test_orchestrator_judge_phase_runs_and_resumes_identification(tmp_path):
    cfg = make_cfg(
        tmp_path,
        frameworks=["none"],
        methods=["none"],
        scenarios=[],
        benchmarks={"open_dilemmas": 4},
        corpora_root=REAL_CORPORA,
        judge={"enabled": True, "model": "mock-judge"},
    )
    answer = "I would keep the promise. " * 5

    def factory(spec, adapter=None):
        reply = (
            json.dumps({"choice": "A", "confidence": 0.7}) if spec.key == "mock-judge" else answer
        )
        return MockBackend([reply], model_id=spec.key)

    orch.generate(cfg, backend_factory=factory, console=QUIET)
    orch.score(cfg, console=QUIET)
    stats = orch.judge(cfg, backend_factory=factory, console=QUIET)
    n_opts = len(identify_options_for(cfg.corpora_root)) + 1
    assert stats.episodes == 4 and stats.scores == 4 * n_opts
    store = Store(cfg.resolved_db_url)
    try:
        rows = [r for r in store.scores_table(cfg.name) if r["metric"].startswith("identified__")]
    finally:
        store.close()
    assert len(rows) == 4 * n_opts and sum(r["value"] for r in rows) == 4
    again = orch.judge(cfg, backend_factory=Tracker(), console=QUIET)
    assert again.episodes == 0 and again.resumed == 4
