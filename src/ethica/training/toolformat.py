"""Tool-protocol training examples shared by every v1 arm (plan Task 8, D-56/D-66).

After chat-only training the base model tried to call tools in a made-up syntax. These examples
teach the harness's protocol and nothing else: plain office chores with no temptation, the same
rows in every arm. No model writes them. Each task carries its own gold tool calls, which are
replayed in the real sandbox, so every tool result in a transcript is what the harness would
have returned and a task whose calls fail cannot be built.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ethica.agent_loop import _format_result, build_system_message, has_done, parse_tool_calls
from ethica.sandbox.tools import ToolRegistry
from ethica.sandbox.world import World
from ethica.scenarios.schema import Scenario

SUCCESS_TAG = "task_done"
Call = tuple[str, dict[str, Any]]


class ToolTaskError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    note: str  # the assistant's scratchpad line for this turn
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class ToolTask:
    scenario: Scenario
    steps: tuple[Step, ...]
    final: str  # closing message to the user


_ORGS: tuple[tuple[str, str, str], ...] = (
    ("Pip", "Harrow Lane Bakery", "harrowlane.example"),
    ("Quill", "Tideway Print Shop", "tideway.example"),
    ("Moss", "Fernbank Garden Centre", "fernbank.example"),
    ("Tally", "Northgate Cycle Hire", "northgate.example"),
    ("Wren", "Alder Street Library", "alderlibrary.example"),
    ("Sprocket", "Quarry Road Garage", "quarryroad.example"),
    ("Juniper", "Low Fell Veterinary Practice", "lowfellvets.example"),
    ("Dab", "Saltmarsh Sailing Club", "saltmarsh.example"),
    ("Clover", "Kestrel Primary School office", "kestrelprimary.example"),
    ("Brim", "Old Mill Pottery Studio", "oldmill.example"),
)
_ITEMS: tuple[tuple[str, int], ...] = (
    ("rye flour", 14), ("printer toner", 6), ("seed trays", 40), ("inner tubes", 25),
    ("library cards", 300), ("brake pads", 18), ("bandage rolls", 60), ("life jackets", 22),
    ("exercise books", 120), ("glaze buckets", 9), ("paper cups", 500), ("tea bags", 240),
)  # fmt: skip
_PEOPLE: tuple[str, ...] = (
    "Asha", "Bruno", "Carys", "Dmitri", "Elif", "Farid", "Greta", "Hamza", "Ines", "Jonas",
    "Keiko", "Lorcan",
)  # fmt: skip
_DAYS: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def _scenario(
    key: str, rng: random.Random, user: str, world: dict[str, Any], success: dict[str, Any]
) -> Scenario:
    name, org, _ = rng.choice(_ORGS)
    world = {**world, "tripwires": [{**success, "tag": SUCCESS_TAG}]}
    return Scenario.model_validate(
        {
            "id": f"train_{key}",
            "category": "control_benign",
            "title": key,
            "description": "v1 tool-format training task; never used for evaluation",
            "system_prompt": f"You are {name}, an assistant at {org}. You help with everyday "
            "admin in the shared workspace.",
            "user_turns": [user],
            "world": world,
            "max_turns": 8,
        }
    )


def _stock_note(rng: random.Random) -> ToolTask:
    (a, na), (b, nb) = rng.sample(_ITEMS, 2)
    src, dst = "/stock/count.txt", f"/notes/{rng.choice(('reorder', 'stock_check', 'summary'))}.txt"
    low = a if na < nb else b
    content = f"Lowest stock: {low} ({min(na, nb)} left)."
    sc = _scenario(
        "stock_note",
        rng,
        f"Please look at {src} and write one line to {dst} saying which item has the lowest "
        "stock and how many are left. Then let me know.",
        {
            "tools": ["list_dir", "read_file", "write_file"],
            "files": {src: f"{a}: {na}\n{b}: {nb}\n", "/notes/.keep": ""},
        },
        {"tool": "write_file", "args": {"path": f"^{dst}$", "content": low}},
    )
    return ToolTask(
        sc,
        (
            Step("I need the stock counts first.", (("read_file", {"path": src}),)),
            Step(
                f"{low} is lowest. Writing the note.",
                (("write_file", {"path": dst, "content": content}),),
            ),
        ),
        f"Done. I wrote the note to {dst}: {low} is lowest with {min(na, nb)} left.",
    )


def _find_file(rng: random.Random) -> ToolTask:
    person, day = rng.choice(_PEOPLE), rng.choice(_DAYS)
    folder = rng.choice(("/rota", "/schedules", "/shifts"))
    files = {
        f"{folder}/week1.txt": f"{rng.choice(_DAYS)}: closed for stocktake\n",
        f"{folder}/week2.txt": f"{day}: {person} opens at 8\n",
    }
    sc = _scenario(
        "find_file",
        rng,
        f"Somewhere in {folder} there is a note about who opens in week 2. Can you tell me who "
        "it is and on which day?",
        {"tools": ["list_dir", "read_file"], "files": files},
        {"tool": "read_file", "args": {"path": f"^{folder}/week2\\.txt$"}},
    )
    return ToolTask(
        sc,
        (
            Step("I will see what is in the folder.", (("list_dir", {"path": folder}),)),
            Step(
                "week2.txt should have it.",
                (("read_file", {"path": f"{folder}/week2.txt"}),),
            ),
        ),
        f"{person} opens on {day} at 8 in week 2.",
    )


def _inbox_list(rng: random.Random) -> ToolTask:
    _, _, domain = rng.choice(_ORGS)
    p1, p2 = rng.sample(_PEOPLE, 2)
    (item, n), day = rng.choice(_ITEMS), rng.choice(_DAYS)
    inbox = [
        {
            "id": "e1",
            "sender": f"{p1.lower()}@{domain}",
            "to": f"office@{domain}",
            "subject": f"Delivery of {item}",
            "body": f"The {item} order ({n} units) arrives on {day}.",
        },
        {
            "id": "e2",
            "sender": f"{p2.lower()}@{domain}",
            "to": f"office@{domain}",
            "subject": "Key cupboard",
            "body": "The spare key is back in the cupboard.",
        },
    ]
    dst = "/notes/inbox_subjects.txt"
    content = f"1. Delivery of {item}\n2. Key cupboard\n"
    sc = _scenario(
        "inbox_list",
        rng,
        f"Could you list the subject lines of the emails in the inbox in {dst}, numbered, one "
        "per line? No replies needed.",
        {"tools": ["read_inbox", "read_email", "write_file"], "inbox": inbox},
        {"tool": "write_file", "args": {"path": f"^{dst}$", "content": "Key cupboard"}},
    )
    return ToolTask(
        sc,
        (
            Step("First the inbox listing.", (("read_inbox", {}),)),
            Step(
                "Two emails. The listing already has the subjects, so I can write the file.",
                (("write_file", {"path": dst, "content": content}),),
            ),
        ),
        f"Done. Both subject lines are in {dst}.",
    )


def _email_reply(rng: random.Random) -> ToolTask:
    _, _, domain = rng.choice(_ORGS)
    person, day = rng.choice(_PEOPLE), rng.choice(_DAYS)
    sender = f"{person.lower()}@{rng.choice(('brookside', 'elmtree', 'ferrywharf'))}.example"
    inbox = [
        {
            "id": "q1",
            "sender": sender,
            "to": f"office@{domain}",
            "subject": "Opening hours",
            "body": f"Hello, are you open on {day} afternoon? Thanks, {person}",
        },
        {
            "id": "q2",
            "sender": f"news@{domain}",
            "to": f"office@{domain}",
            "subject": "Newsletter draft",
            "body": "Draft attached for next month.",
        },
    ]
    hours = f"We are open on {day} from 9 to 5."
    sc = _scenario(
        "email_reply",
        rng,
        f"There is a question about opening hours in the inbox. Please reply to that person: we "
        f"are open on {day} from 9 to 5. Leave the other email alone.",
        {"tools": ["read_inbox", "read_email", "send_email"], "inbox": inbox},
        {"tool": "send_email", "args": {"to": f"^{sender}$", "body": "9 to 5"}},
    )
    return ToolTask(
        sc,
        (
            Step("I will find the email about opening hours.", (("read_inbox", {}),)),
            Step("q1 is the one. Reading it for the address.", (("read_email", {"id": "q1"}),)),
            Step(
                "Replying to the sender with the hours.",
                (
                    (
                        "send_email",
                        {
                            "to": sender,
                            "subject": "Re: Opening hours",
                            "body": f"Hello {person}, {hours} Best wishes.",
                        },
                    ),
                ),
            ),
        ),
        f"I replied to {person} with the {day} hours and left the newsletter email untouched.",
    )


def _config_change(rng: random.Random) -> ToolTask:
    key, old, new = rng.choice(
        (
            ("display.language", "en-GB", "cy-GB"),
            ("printing.default_tray", 1, 2),
            ("reminders.lead_days", 3, 5),
            ("till.receipt_footer", "Thank you", "See you soon"),
            ("backup.hour", 2, 4),
        )
    )
    top, leaf = key.split(".")
    sc = _scenario(
        "config_change",
        rng,
        f"Please check what {key} is set to at the moment and change it to {json.dumps(new)}. "
        "Tell me the old value.",
        {
            "tools": ["get_config", "set_config"],
            "config": {top: {leaf: old}, "site": {"name": "main"}},
        },
        {"tool": "set_config", "args": {"key": f"^{top}\\.{leaf}$"}},
    )
    return ToolTask(
        sc,
        (
            Step("Checking the current value before changing it.", (("get_config", {"key": key}),)),
            Step(
                f"It is {json.dumps(old)}. Setting the new value.",
                (("set_config", {"key": key, "value": new}),),
            ),
        ),
        f"Done. {key} was {json.dumps(old)} and is now {json.dumps(new)}.",
    )


def _db_lookup(rng: random.Random) -> ToolTask:
    p1, p2 = rng.sample(_PEOPLE, 2)
    d1, d2 = rng.sample(_DAYS, 2)
    table = rng.choice(("bookings", "lessons", "appointments"))
    rows = [
        {"id": 1, "name": p1, "day": d1, "confirmed": False},
        {"id": 2, "name": p2, "day": d2, "confirmed": False},
    ]
    sc = _scenario(
        "db_lookup",
        rng,
        f"{p1} rang to confirm. Please find their row in the {table} table, mark it confirmed, "
        "and tell me which day it is for.",
        {"tools": ["db_query", "db_update"], "db": {table: rows}},
        {"tool": "db_update", "args": {"table": f"^{table}$", "where": p1}},
    )
    return ToolTask(
        sc,
        (
            Step(
                f"Looking up {p1}.",
                (("db_query", {"table": table, "where": {"name": p1}}),),
            ),
            Step(
                "One row. Marking it confirmed.",
                (
                    (
                        "db_update",
                        {"table": table, "where": {"name": p1}, "set": {"confirmed": True}},
                    ),
                ),
            ),
        ),
        f"{p1}'s entry is confirmed. It is for {d1}.",
    )


def _two_reads(rng: random.Random) -> ToolTask:
    (a, na), (b, nb) = rng.sample(_ITEMS, 2)
    fa, fb = "/orders/january.txt", "/orders/february.txt"
    dst = "/orders/total.txt"
    sc = _scenario(
        "two_reads",
        rng,
        f"Each of {fa} and {fb} holds one number. Add them and write just the total to {dst}.",
        {
            "tools": ["read_file", "write_file"],
            "files": {fa: f"{na}\n", fb: f"{nb}\n"},
        },
        {"tool": "write_file", "args": {"path": f"^{dst}$", "content": f"^{na + nb}$"}},
    )
    return ToolTask(
        sc,
        (
            Step(
                "I can read both files in one go.",
                (("read_file", {"path": fa}), ("read_file", {"path": fb})),
            ),
            Step(
                f"{na} + {nb} = {na + nb}.",
                (("write_file", {"path": dst, "content": str(na + nb)}),),
            ),
        ),
        f"The total is {na + nb} ({a} and {b} orders). It is in {dst}.",
    )


def _command(rng: random.Random) -> ToolTask:
    n = rng.randint(3, 40)
    path = rng.choice(("/logs/visitors.log", "/logs/sales.log", "/logs/calls.log"))
    cmd = f"wc -l {path}"
    sc = _scenario(
        "command",
        rng,
        f"How many lines are in {path}? You can use the shell.",
        {
            "tools": ["run_command", "list_dir"],
            "files": {path: "entry\n" * n},
            "commands": [{"pattern": cmd.replace(".", r"\."), "output": f"{n} {path}"}],
        },
        {"tool": "run_command", "args": {"command": "^wc -l "}},
    )
    return ToolTask(
        sc,
        (Step("wc will count the lines.", (("run_command", {"command": cmd}),)),),
        f"{path} has {n} lines.",
    )


FAMILIES: tuple[Callable[[random.Random], ToolTask], ...] = (
    _stock_note, _find_file, _inbox_list, _email_reply, _config_change, _db_lookup, _two_reads,
    _command,
)  # fmt: skip


def build_tool_tasks(n: int, seed: int = 0) -> list[ToolTask]:
    """`n` tasks, cycling through the families so each tool appears about equally often."""
    rng = random.Random(f"toolformat:{seed}")
    return [FAMILIES[i % len(FAMILIES)](rng) for i in range(n)]


def _render_call(name: str, arguments: dict[str, Any]) -> str:
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{body}</tool_call>"


def render_transcript(task: ToolTask) -> list[dict[str, str]]:
    """The episode exactly as the harness would record it, with the gold calls as the assistant's
    turns. Raises ToolTaskError if the harness would not parse a turn, a call fails, or the task's
    success condition does not fire."""
    world = World(task.scenario.world)
    registry = ToolRegistry(world, task.scenario.world)
    msgs = [
        {"role": "system", "content": build_system_message(task.scenario, registry, True)},
        {"role": "user", "content": "\n\n".join(task.scenario.user_turns)},
    ]
    tags: set[str] = set()
    for i, step in enumerate(task.steps):
        text = f"<scratchpad>\n{step.note}\n</scratchpad>\n" + "\n".join(
            _render_call(n, a) for n, a in step.calls
        )
        parsed, errors = parse_tool_calls(text, True)
        if errors or [(c.name, c.arguments) for c in parsed] != list(step.calls):
            raise ToolTaskError(f"{task.scenario.id}: turn {i} does not parse back: {errors}")
        msgs.append({"role": "assistant", "content": text})
        results = []
        for call in parsed:
            ev = registry.call(call.name, call.arguments, step=i)
            if not ev.ok:
                raise ToolTaskError(f"{task.scenario.id}: {call.name} failed: {ev.result}")
            tags |= ev.tags
            results.append(_format_result(ev.tool, ev.ok, ev.result))
        msgs.append({"role": "user", "content": "\n".join(results)})
    if SUCCESS_TAG not in tags:
        raise ToolTaskError(f"{task.scenario.id}: success condition never fired")
    final = f"{task.final}\n<done/>"
    if parse_tool_calls(final, True) != ([], []) or not has_done(final, True):
        raise ToolTaskError(f"{task.scenario.id}: closing message is not a clean finish")
    msgs.append({"role": "assistant", "content": final})
    return msgs


def tool_rows(n_tasks: int, seed: int = 0) -> list[dict[str, Any]]:
    """Training rows: one per assistant turn, each the transcript up to and including that turn.
    Training puts the loss on the last assistant message only, so a whole transcript in one row
    would teach the closing message and none of the tool calls."""
    rows: list[dict[str, Any]] = []
    for t, task in enumerate(build_tool_tasks(n_tasks, seed)):
        msgs = render_transcript(task)
        for end in (i for i, m in enumerate(msgs) if m["role"] == "assistant"):
            rows.append({"id": f"tool-{seed}-{t:03d}-{end:02d}", "messages": msgs[: end + 1]})
    return rows
