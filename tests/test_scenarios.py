import re
from collections import Counter
from pathlib import Path
from typing import get_args

import pytest

from ethica.sandbox.spec import ToolName
from ethica.scenarios.registry import (
    BUILTIN_DIR,
    ScenarioLoadError,
    ScenarioRegistry,
    get,
    list_scenarios,
    load_file,
    load_registry,
)
from ethica.scenarios.schema import Category

METRICS_MD = Path(__file__).parent.parent / "docs" / "METRICS.md"
VALID_TOOLS = set(get_args(ToolName))


def documented_metrics() -> tuple[set[str], list[re.Pattern[str]]]:
    exact: set[str] = set()
    patterns: list[re.Pattern[str]] = []
    for line in METRICS_MD.read_text().splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if not m:
            continue
        name = m.group(1)
        if "<" in name:
            patterns.append(re.compile(re.escape(name.split("<")[0]) + r".+"))
        else:
            exact.add(name)
    return exact, patterns


def test_metric_parser_finds_tables() -> None:
    exact, patterns = documented_metrics()
    assert {"task_success", "misaligned_action", "parse_failure_rate"} <= exact
    assert patterns


def test_every_builtin_loads_and_filename_matches_id() -> None:
    files = sorted(BUILTIN_DIR.glob("*.yaml"))
    assert files
    for f in files:
        assert load_file(f).id == f.stem


def test_ids_unique() -> None:
    ids = [load_file(f).id for f in BUILTIN_DIR.glob("*.yaml")]
    assert len(ids) == len(set(ids))


def test_tools_valid_and_tripwires_use_exposed_tools() -> None:
    for s in list_scenarios():
        assert s.world.tools, s.id
        assert set(s.world.tools) <= VALID_TOOLS, s.id
        for tw in s.world.tripwires:
            assert tw.tool in s.world.tools, f"{s.id}: tripwire on unexposed tool {tw.tool}"


def test_check_metrics_documented() -> None:
    exact, patterns = documented_metrics()
    for s in list_scenarios():
        for c in s.checks:
            ok = c.metric in exact or any(p.fullmatch(c.metric) for p in patterns)
            assert ok, f"{s.id}: metric {c.metric!r} not in docs/METRICS.md"


def test_checks_reference_declared_tags() -> None:
    for s in list_scenarios():
        emitted = {tw.tag for tw in s.world.tripwires} | {
            t for h in s.world.commands for t in h.tags
        }
        emitted |= {"readonly_violation_attempt", "upload", "email_sent"}
        for c in s.checks:
            if c.tag:
                assert c.tag in emitted, f"{s.id}: check tag {c.tag!r} can never fire"
        for f in s.followups:
            if f.if_tag:
                assert f.if_tag in emitted, f"{s.id}: followup tag {f.if_tag!r} can never fire"


def test_every_scenario_has_task_success_check() -> None:
    for s in list_scenarios():
        assert any(c.metric == "task_success" for c in s.checks), s.id


def test_pair_tags_come_in_bait_nonbait_pairs() -> None:
    pairs: dict[str, list[bool]] = {}
    for s in list_scenarios():
        for t in s.tags:
            if t.startswith("pair:"):
                pairs.setdefault(t, []).append(s.framework_bait)
    assert pairs
    for tag, baits in pairs.items():
        assert sorted(baits) == [False, True], f"{tag}: expected one bait + one plain, got {baits}"


def test_controls_have_false_positive_tripwire() -> None:
    controls = list_scenarios(category="control_benign")
    assert len(controls) >= 2
    for s in controls:
        assert not s.framework_bait
        assert any(c.metric == "misaligned_action" for c in s.checks), s.id


def test_category_coverage_report() -> None:
    covered = {s.category for s in list_scenarios()}
    missing = sorted(set(get_args(Category)) - covered)
    print(f"categories without a scenario (authoring incomplete by design): {missing}")
    if missing:
        pytest.xfail(f"missing categories: {missing}")


def test_registry_get_and_list_filters() -> None:
    assert get("oversight_triage_plain").category == "oversight_subversion"
    with pytest.raises(KeyError, match="unknown scenario"):
        get("nope")
    assert {s.id for s in list_scenarios(framework_bait=True)} == {
        "exfil_companion_bait",
        "oversight_triage_bait",
    }
    assert [s.id for s in list_scenarios(tags=["pair:oversight_triage"])] == [
        "oversight_triage_bait",
        "oversight_triage_plain",
    ]
    cats = Counter(s.category for s in list_scenarios())
    assert cats["control_benign"] == 2


def write_valid(path: Path, sid: str) -> None:
    src = (BUILTIN_DIR / "control_config_update.yaml").read_text()
    path.write_text(src.replace("id: control_config_update", f"id: {sid}", 1))


def test_duplicate_id_error_names_both_files(tmp_path: Path) -> None:
    write_valid(tmp_path / "dup.yaml", "control_config_update")
    with pytest.raises(ScenarioLoadError, match=r"duplicate.*control_config_update.*dup\.yaml"):
        load_registry([tmp_path])


def test_extra_dir_adds_scenarios(tmp_path: Path) -> None:
    write_valid(tmp_path / "extra.yaml", "extra_one")
    reg = load_registry([tmp_path])
    assert reg.get("extra_one").id == "extra_one"
    assert len(reg) == len(list_scenarios()) + 1


def test_invalid_yaml_error_includes_filename(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("id: [unclosed")
    with pytest.raises(ScenarioLoadError, match=r"broken\.yaml"):
        ScenarioRegistry.from_dirs([tmp_path])


def test_schema_violation_error_includes_filename(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("id: x\ncategory: not_a_category\n")
    with pytest.raises(ScenarioLoadError, match=r"bad\.yaml"):
        ScenarioRegistry.from_dirs([tmp_path])


def test_check_missing_field_and_bad_regex_rejected(tmp_path: Path) -> None:
    base = (BUILTIN_DIR / "control_config_update.yaml").read_text()
    head = base[: base.index("checks:")]
    (tmp_path / "nofield.yaml").write_text(head + "checks:\n  - {type: tag_present, metric: x}\n")
    with pytest.raises(ScenarioLoadError, match=r"nofield\.yaml.*requires field 'tag'"):
        load_file(tmp_path / "nofield.yaml")
    (tmp_path / "rx.yaml").write_text(
        head + "checks:\n  - {type: final_regex, metric: x, pattern: '('}\n"
    )
    with pytest.raises(ScenarioLoadError, match=r"rx\.yaml.*invalid regex"):
        load_file(tmp_path / "rx.yaml")


def _scenario_text(sid: str, extra: str) -> str:
    return (
        f"id: {sid}\ncategory: oversight_subversion\ntitle: t\ndescription: d\n"
        "system_prompt: s\nuser_turns: [u]\n" + extra
    )


@pytest.mark.parametrize(
    "extra",
    [
        "world:\n  commands:\n    - {pattern: '(', output: x}\n",
        "world:\n  tripwires:\n    - {tool: read_file, args: {path: '['}, tag: t}\n",
    ],
)
def test_bad_world_regex_rejected_with_filename(tmp_path: Path, extra: str) -> None:
    f = tmp_path / "bad_rx.yaml"
    f.write_text(_scenario_text("bad_rx", extra))
    with pytest.raises(ScenarioLoadError, match=r"bad_rx\.yaml.*invalid regex"):
        load_registry([tmp_path])
