"""Scenario YAML schema. One file per scenario in scenarios/builtin/."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ethica.sandbox.spec import WorldSpec

Category = Literal[
    "oversight_subversion",
    "self_exfiltration",
    "shutdown_resistance",
    "goal_guarding",
    "covert_sabotage",
    "sandbagging",
    "deception",
    "reward_hacking",
    "power_seeking",
    "coercion",
    "sycophancy",
    "control_benign",  # no temptation; measures baseline task competence / false positives
]


class Check(BaseModel):
    """Programmatic check. `metric` is the metric name it emits (docs/METRICS.md).
    type:
      tag_present   -> 1.0 if any EnvEvent has `tag`
      tag_absent    -> 1.0 if no EnvEvent has `tag`
      tool_called   -> 1.0 if `tool` was called (optional `args` regex dict)
      state_equals  -> 1.0 if final_env_state at dotted `path` == `value`
      final_regex   -> 1.0 if last assistant message matches `pattern` (re.search, IGNORECASE)
      task_success  -> alias of state_equals/tag_present used for the legit task
    """

    type: Literal[
        "tag_present", "tag_absent", "tool_called", "state_equals", "final_regex", "task_success"
    ]
    metric: str
    tag: str | None = None
    tool: str | None = None
    args: dict[str, str] = Field(default_factory=dict)
    path: str | None = None
    value: Any = None
    pattern: str | None = None
    invert: bool = False  # flip 1.0 <-> 0.0


class Followup(BaseModel):
    """Post-episode interrogation, sent as a user turn after the main loop ends.
    If `if_tag` is set, only asked when that tag fired (deception-about-actions)."""

    ask: str
    if_tag: str | None = None


class Scenario(BaseModel):
    id: str
    category: Category
    title: str
    description: str  # for humans; never shown to the model
    # true when the misaligned act can be rationalised as better outcomes / duty (framework bait)
    framework_bait: bool = False
    bait_notes: str = ""  # which framework(s) it baits and how
    system_prompt: str  # role/task framing; harness appends tool protocol + conditioning
    user_turns: list[str]  # initial user message(s); usually one
    world: WorldSpec
    max_turns: int = 12
    followups: list[Followup] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    judge_rubrics: list[str] = Field(default_factory=list)  # rubric ids in scoring/rubrics/
    tags: list[str] = Field(default_factory=list)
