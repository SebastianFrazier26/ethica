"""Load and validate scenario YAMLs. Builtins live next to this module in `builtin/`."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from ethica.scenarios.schema import Check, Scenario

BUILTIN_DIR = Path(__file__).parent / "builtin"

# Which Check field each type needs; scoring would otherwise fail per-episode, not at load.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "tag_present": ("tag",),
    "tag_absent": ("tag",),
    "tool_called": ("tool",),
    "state_equals": ("path",),
    "final_regex": ("pattern",),
}


class ScenarioLoadError(ValueError):
    """A scenario file is unreadable, malformed, or inconsistent with the registry."""


def _check_error(check: Check) -> str | None:
    if check.type == "task_success":
        if not any((check.tag, check.tool, check.path, check.pattern)):
            return "task_success needs one of tag/tool/path/pattern"
        required: tuple[str, ...] = ()
    else:
        required = _REQUIRED[check.type]
    for name in required:
        if getattr(check, name) is None:
            return f"check type {check.type!r} requires field {name!r}"
    patterns = [check.pattern, *check.args.values()] if check.pattern else [*check.args.values()]
    for p in patterns:
        try:
            re.compile(p)
        except re.error as e:
            return f"invalid regex {p!r}: {e}"
    return None


def load_file(path: Path) -> Scenario:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ScenarioLoadError(f"{path}: cannot read/parse YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ScenarioLoadError(f"{path}: top level must be a mapping")
    try:
        scenario = Scenario.model_validate(raw)
    except ValidationError as e:
        raise ScenarioLoadError(f"{path}: invalid scenario: {e}") from e
    for i, check in enumerate(scenario.checks):
        if (err := _check_error(check)) is not None:
            raise ScenarioLoadError(f"{path}: checks[{i}] ({check.metric}): {err}")
    for i, handler in enumerate(scenario.world.commands):
        _compile_or_raise(path, f"world.commands[{i}].pattern", handler.pattern)
    for i, tw in enumerate(scenario.world.tripwires):
        for arg, pattern in tw.args.items():
            _compile_or_raise(path, f"world.tripwires[{i}].args[{arg!r}]", pattern)
    return scenario


def _compile_or_raise(path: Path, where: str, pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ScenarioLoadError(f"{path}: {where}: invalid regex {pattern!r}: {e}") from e


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ScenarioLoadError(f"{directory}: not a directory")
    return sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])


class ScenarioRegistry:
    def __init__(self, scenarios: Iterable[tuple[Path, Scenario]] = ()) -> None:
        self._by_id: dict[str, Scenario] = {}
        self._source: dict[str, Path] = {}
        for path, scenario in scenarios:
            if scenario.id in self._by_id:
                raise ScenarioLoadError(
                    f"duplicate scenario id {scenario.id!r}: {self._source[scenario.id]} and {path}"
                )
            self._by_id[scenario.id] = scenario
            self._source[scenario.id] = path

    @classmethod
    def from_dirs(cls, directories: Sequence[Path]) -> ScenarioRegistry:
        return cls((p, load_file(p)) for d in directories for p in _yaml_files(d))

    def get(self, scenario_id: str) -> Scenario:
        try:
            return self._by_id[scenario_id]
        except KeyError:
            known = ", ".join(sorted(self._by_id)) or "none"
            raise KeyError(f"unknown scenario {scenario_id!r}. Known: {known}") from None

    def list(
        self,
        category: str | None = None,
        framework_bait: bool | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[Scenario]:
        """Sorted by id. `tags` requires all given tags to be present."""
        want = set(tags or ())
        return [
            s
            for _, s in sorted(self._by_id.items())
            if (category is None or s.category == category)
            and (framework_bait is None or s.framework_bait == framework_bait)
            and want <= set(s.tags)
        ]

    def source_of(self, scenario_id: str) -> Path:
        return self._source[scenario_id]

    def __len__(self) -> int:
        return len(self._by_id)


@lru_cache(maxsize=1)
def _builtin() -> ScenarioRegistry:
    return ScenarioRegistry.from_dirs([BUILTIN_DIR])


def load_registry(extra_dirs: Sequence[Path] = ()) -> ScenarioRegistry:
    """Builtins plus any extra directories; duplicate ids across them are an error."""
    if not extra_dirs:
        return _builtin()
    return ScenarioRegistry.from_dirs([BUILTIN_DIR, *extra_dirs])


def get(scenario_id: str) -> Scenario:
    return _builtin().get(scenario_id)


def list_scenarios(
    category: str | None = None,
    framework_bait: bool | None = None,
    tags: Iterable[str] | None = None,
) -> list[Scenario]:
    return _builtin().list(category, framework_bait, tags)
