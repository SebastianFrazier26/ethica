"""Critique and revise for v1 framework arms (proposal §3, option D3; plan Task 9).

Each kept answer of an ethics arm is checked by the teacher against the arm's commitments
statement and rewritten where it departs from them. The output file has the same row schema as
`generate.py`, so `dataset.py` builds training files from it unchanged. Neutral prompts and
rows already rejected pass through untouched: format skills and the shared prompt set must not
differ between arms.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ethica.training.generate import _STYLE, _VOICE, ArmSpec, _prompt_seed, read_rows, screen
from ethica.types import ChatBackend, GenerationParams, Message

NO_DEPARTURE = "NONE"

CRITIC_INSTRUCTION = (
    "You check an answer against a statement of commitments. List each place where the "
    "answer's verdict or its reasons depart from the commitments, one per line, quoting the "
    f"commitment it departs from. If it departs from none, reply with the single word "
    f"{NO_DEPARTURE}. Do not judge style, length or tone."
)
REVISE_INSTRUCTION = (
    _VOICE + "Rewrite the draft answer so that it follows from the commitments, fixing the "
    "problems listed in the critique and keeping everything else. Do not mention the "
    "commitments, the critique or the draft, and do not name any school of thought. " + _STYLE
)


def critique_messages(commitments: str, prompt: str, answer: str) -> list[Message]:
    return [
        Message("system", CRITIC_INSTRUCTION),
        Message(
            "user",
            f"Commitments:\n{commitments.strip()}\n\nQuestion:\n{prompt}\n\nAnswer:\n{answer}",
        ),
    ]


def revise_messages(commitments: str, prompt: str, answer: str, critique: str) -> list[Message]:
    return [
        Message("system", "These are commitments you hold:\n" + commitments.strip()),
        Message(
            "user",
            f"{REVISE_INSTRUCTION}\n\nQuestion:\n{prompt}\n\nDraft answer:\n{answer}\n\n"
            f"Critique:\n{critique}",
        ),
    ]


def no_departure(critique: str) -> bool:
    return critique.strip().strip(".").upper() == NO_DEPARTURE


@dataclass
class ReviseStats:
    critiqued: int = 0
    revised: int = 0
    unchanged: int = 0
    passed_through: int = 0
    skipped_existing: int = 0
    rejected: dict[str, int] = field(default_factory=dict)


def revise_arm(
    backend: ChatBackend,
    arm: ArmSpec,
    answers_path: Path,
    out_path: Path,
    params: GenerationParams,
) -> ReviseStats:
    """Write `out_path` from `answers_path`; resumes from rows already in `out_path`."""
    if arm.kind != "ethics" or not arm.commitments:
        raise ValueError(f"arm {arm.name!r}: critique and revise needs signed-off commitments")
    source = read_rows(answers_path)
    existing = read_rows(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing), encoding="utf-8"
    )
    done = {r["prompt_id"] for r in existing}
    stats = ReviseStats(skipped_existing=len(done))
    with out_path.open("a", encoding="utf-8") as fh:
        for row in source:
            if row["prompt_id"] in done:
                continue
            out: dict[str, Any] = dict(row)
            if row["reject"] is not None or row["category"] == "neutral":
                out["revision"] = "passthrough"
                stats.passed_through += 1
            else:
                # different seeds for the two calls; both fixed per prompt so a resume matches
                seed = _prompt_seed(row["prompt_id"])
                crit = backend.generate(
                    critique_messages(arm.commitments, row["prompt"], row["answer"]),
                    replace(params, seed=seed, temperature=0.0),
                ).text.strip()
                stats.critiqued += 1
                out["critique"] = crit
                if no_departure(crit):
                    out["revision"] = "unchanged"
                    stats.unchanged += 1
                else:
                    c = backend.generate(
                        revise_messages(arm.commitments, row["prompt"], row["answer"], crit),
                        replace(params, seed=seed + 1),
                    )
                    out["original_answer"] = row["answer"]
                    out["answer"] = c.text.strip()
                    out["finish_reason"] = c.finish_reason
                    out["reject"] = screen(c.text, c.finish_reason, arm)
                    out["revision"] = "revised"
                    stats.revised += 1
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
    stats.rejected = dict(
        Counter(r["reject"] for r in read_rows(out_path) if r["reject"] is not None)
    )
    return stats
