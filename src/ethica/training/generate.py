"""Teacher answers for one v1 training arm (proposal §3, D-64).

A local teacher model answers the shared prompt set. For framework and placebo arms the arm's
corpus is retrieved into context exactly as in v0's RAG method; the generic arm gets nothing.
Rows are appended to a JSONL file as they are produced, so an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ethica.scoring.judge import FRAMEWORK_KEY_TERMS
from ethica.training.prompts import TrainPrompt
from ethica.types import ChatBackend, Conditioner, GenerationParams, Message

# "You"/"I" are spelled out because the first wording ("answer in the first person, in your own
# voice") made the teacher answer as the person asking: "I should talk to my flatmate"
# (feasibility check, 2026-11-23).
_VOICE = (
    'The user is asking you. Reply to them directly: say "you" for the user, and "I" only '
    "for your own views. "
)
_STYLE = "Be direct and specific. Keep the answer under 250 words."
GENERIC_INSTRUCTION = _VOICE + _STYLE
OUTLOOK_INSTRUCTION = (
    _VOICE + "The outlook in the background reading is your own. Do not mention the reading, "
    "do not name any school of thought, and do not say you were asked to take a view. " + _STYLE
)
COMMITMENTS_HEADER = "These are commitments you hold, in your own words:\n"
# Owner, 2026-11-24: with the first-person commitments placed right before the instruction, 20-44%
# of ethics-arm answers spoke as the person asking ("I should tell the client"), against 1% in the
# controls (docs/study-v1/PILOT-DATA-2026-11-24.md). Only ethics arms get this sentence; the
# controls' instruction is unchanged, so their stored answers stay valid.
COMMITMENTS_VOICE = (
    "They describe how you yourself reason. The user is the one facing the situation: advise "
    'them on what they should do and address them as "you". Never answer as if you were the '
    'user or in their place; use "I" only to give your own view, as in "I think you should".'
)

# Answers that decline the user's request. Narrower than the benchmark heuristic
# (`extract.is_refusal`) on purpose (owner, 2026-11-24): that one matched "I can't / I won't" and
# so screened out principled answers ("I won't lie to her"), 31 of them in the deontology arm and
# 5-7 in the others; every one of those was a false positive.
_DECLINE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:can't|cannot|can not|won't|will not|unable to|not able to) (?:assist|help)"
        r"(?: you)?(?: with)? (?:that|this|your|such|the request|it\b)",
        r"\b(?:can't|cannot|can not|won't|will not|unable to|not able to) "
        r"(?:provide|comply|fulfil|fulfill|engage|generate|create|write|give)(?: you)? "
        r"(?:that|this|with|such|the requested|your|any|information|instructions|guidance|"
        r"help|assistance)",
        r"\bi(?:'m| am) sorry,? but i (?:can't|cannot|won't|will not|am unable|'m unable|"
        r"am not able)",
        r"\bi (?:must|have to|need to) (?:respectfully )?decline\b",
        r"\bagainst my (?:guidelines|policy|policies|programming)",
        r"\b(?:not|isn't|is not) (?:something|a request) i (?:can|will|am able to) (?:help|assist)",
    )
]


def declines_request(text: str) -> bool:
    return any(p.search(text) for p in _DECLINE)


_META = re.compile(r"background reading|excerpt|the reading\b|as an ai\b", re.IGNORECASE)
MIN_CHARS = 20


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: Literal["ethics", "placebo", "generic"]
    corpus: str | None = None
    # owner-approved statement of the framework's commitments (D-64); None until signed off
    commitments: str | None = None

    def __post_init__(self) -> None:
        if (self.kind == "generic") != (self.corpus is None):
            raise ValueError(f"arm {self.name!r}: generic arms have no corpus, others need one")


def teacher_messages(
    arm: ArmSpec, prompt: TrainPrompt, conditioner: Conditioner | None
) -> list[Message]:
    # neutral prompts are answered the same way in every arm: format skills must not differ
    plain = arm.kind == "generic" or prompt.category == "neutral"
    if plain:
        return [Message("system", GENERIC_INSTRUCTION), Message("user", prompt.text)]
    if conditioner is None:
        raise ValueError(f"arm {arm.name!r} needs a conditioner for its corpus")
    system = OUTLOOK_INSTRUCTION
    if arm.commitments:
        system = (
            COMMITMENTS_HEADER + arm.commitments.strip() + "\n\n" + COMMITMENTS_VOICE + "\n\n"
        ) + system
    return conditioner.prepare([Message("system", system), Message("user", prompt.text)])


def screen(text: str, finish_reason: str, arm: ArmSpec | None = None) -> str | None:
    """Reason to keep an answer out of the training data, or None to keep it."""
    if len(text.strip()) < MIN_CHARS:
        return "empty"
    if finish_reason == "length":
        return "truncated"
    if declines_request(text):
        return "refusal"
    if _META.search(text):
        return "meta"
    if arm is not None and arm.kind == "ethics":
        terms = FRAMEWORK_KEY_TERMS.get(arm.name, ())
        if any(re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE) for t in terms):
            return "named"
    return None


@dataclass
class GenStats:
    generated: int = 0
    skipped_existing: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Complete rows only: a final line cut off by an interrupted run is ignored."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def _prompt_seed(prompt_id: str) -> int:
    return int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8], 16)


def generate_arm(
    backend: ChatBackend,
    arm: ArmSpec,
    prompts: Sequence[TrainPrompt],
    out_path: Path,
    params: GenerationParams,
    conditioner: Conditioner | None = None,
) -> GenStats:
    existing = read_rows(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # rewrite what was read: drops a torn last line so the next append starts on a clean row
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing), encoding="utf-8"
    )
    done = {r["prompt_id"] for r in existing}
    stats = GenStats(skipped_existing=sum(1 for p in prompts if p.id in done))
    with out_path.open("a", encoding="utf-8") as fh:
        for p in prompts:
            if p.id in done:
                continue
            # seeded per prompt, not per position: a resumed run gives the rows an
            # uninterrupted one would have
            c = backend.generate(
                teacher_messages(arm, p, conditioner), replace(params, seed=_prompt_seed(p.id))
            )
            row = {
                "prompt_id": p.id,
                "category": p.category,
                "prompt": p.text,
                "answer": c.text.strip(),
                "finish_reason": c.finish_reason,
                "reject": screen(c.text, c.finish_reason, arm),
                "arm": arm.name,
                "teacher": backend.model_id,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            stats.generated += 1
    rows = read_rows(out_path)
    stats.kept = sum(1 for r in rows if r["reject"] is None)
    stats.rejected = dict(Counter(r["reject"] for r in rows if r["reject"] is not None))
    return stats


def rescreen(path: Path, arm: ArmSpec) -> dict[str, int]:
    """Apply the current `screen` to stored answers in place, so a screen change does not need
    the teacher again. The first earlier verdict is kept as `reject_before_rescreen`."""
    rows = read_rows(path)
    for r in rows:
        r.setdefault("reject_before_rescreen", r["reject"])
        r["reject"] = screen(r["answer"], r["finish_reason"], arm)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return dict(Counter(r["reject"] for r in rows if r["reject"] is not None))
