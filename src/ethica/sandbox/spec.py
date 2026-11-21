"""Declarative world spec. Scenarios describe the world + tripwires in YAML; the sandbox
is generic and never hard-codes scenario semantics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal[
    "list_dir",
    "read_file",
    "write_file",
    "delete_file",
    "read_inbox",
    "read_email",
    "send_email",
    "get_config",
    "set_config",
    "run_command",
    "db_query",
    "db_update",
    "upload",
    "submit",
]


class Email(BaseModel):
    id: str
    sender: str
    to: str
    subject: str
    body: str
    read: bool = False


class CommandHandler(BaseModel):
    """Simulated shell: first handler whose regex matches the command wins.
    Unmatched commands return `bash: <cmd>: command not found`."""

    pattern: str  # Python regex, re.fullmatch against the stripped command
    output: str = ""
    tags: list[str] = Field(default_factory=list)
    # optional state mutations applied when matched, e.g. {"config": {"oversight.enabled": false}}
    effects: dict[str, Any] = Field(default_factory=dict)


class Tripwire(BaseModel):
    """Attach semantic tags to tool calls. Matching is on tool name + regex over
    each named argument (stringified). All listed arg patterns must match."""

    tool: ToolName
    args: dict[str, str] = Field(default_factory=dict)  # arg name -> regex (re.search)
    tag: str


class WorldSpec(BaseModel):
    files: dict[str, str] = Field(default_factory=dict)  # abs path -> content
    readonly: list[str] = Field(default_factory=list)  # path prefixes write/delete refuse
    inbox: list[Email] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    db: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # table -> rows
    commands: list[CommandHandler] = Field(default_factory=list)
    tripwires: list[Tripwire] = Field(default_factory=list)
    # tool names exposed to the model for this scenario (subset of ToolName)
    tools: list[ToolName] = Field(default_factory=list)
