"""Framework-fidelity probe: did conditioning make the model answer like framework X?

The expected-answer mappings in data/fidelity.yaml are contestable textbook positions, each
tagged with a confidence. Items where a framework is indeterminate score NaN for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ethica.benchmarks.base import (
    NAN,
    BenchItem,
    ScoreTuple,
    item_rng,
    resolve_n,
    sample_indices,
    unparseable_score,
)
from ethica.benchmarks.extract import extract_letter

FRAMEWORKS = (
    "deontology",
    "act_utilitarian",
    "rule_utilitarian",
    "virtue",
    "divine_command",
    "care",
    "contractualism",
)
DATA_PATH = Path(__file__).parent / "data" / "fidelity.yaml"

Confidence = Literal["high", "med", "low"]


class Expectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str | None
    confidence: Confidence
    rationale: str


class FidelityItem(BaseModel):
    id: str
    topic: str
    prompt: str
    choices: dict[str, str]
    expected: dict[str, Expectation]

    @field_validator("expected", mode="before")
    @classmethod
    def _unpack(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            raise ValueError("expected must be a mapping")
        out: dict[str, Any] = {}
        for fw, triple in v.items():
            if not (isinstance(triple, list | tuple) and len(triple) == 3):
                raise ValueError(f"{fw}: expected [answer, confidence, rationale]")
            out[fw] = {"answer": triple[0], "confidence": triple[1], "rationale": triple[2]}
        return out

    @model_validator(mode="after")
    def _check(self) -> FidelityItem:
        letters = list(self.choices)
        if letters != list("ABC"[: len(letters)]) or len(letters) not in (2, 3):
            raise ValueError(f"{self.id}: choices must be A,B[,C], got {letters}")
        if set(self.expected) != set(FRAMEWORKS):
            raise ValueError(f"{self.id}: expected keys {sorted(self.expected)} != frameworks")
        for fw, e in self.expected.items():
            if e.answer is not None and e.answer not in self.choices:
                raise ValueError(f"{self.id}/{fw}: answer {e.answer!r} not in choices")
            if not e.rationale.strip():
                raise ValueError(f"{self.id}/{fw}: empty rationale")
        return self


@cache
def load_fidelity_items(path: Path = DATA_PATH) -> tuple[FidelityItem, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = tuple(FidelityItem.model_validate(x) for x in raw["items"])
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fidelity item ids")
    return items


_INSTRUCTION = (
    "Choose exactly one option. Reply in this format:\nAnswer: <letter>\nReason: <one sentence>"
)


@dataclass
class FidelityBenchmark:
    default_n: int = 48
    shuffle: bool = True
    system_prompt: str | None = None
    name: str = "fidelity"

    def load(self, n: int | None = None, seed: int = 0) -> list[BenchItem]:
        items = load_fidelity_items()
        out = []
        for i in sample_indices(len(items), resolve_n(n, self.default_n), seed):
            it = items[i]
            order = list(it.choices)
            # authored answers skew to B (~65%); permute options per (seed, item) so position
            # bias in small models doesn't masquerade as framework alignment
            if self.shuffle:
                item_rng(seed, it.id).shuffle(order)
            new_letter = {old: "ABC"[pos] for pos, old in enumerate(order)}
            opts = "\n".join(f"{new_letter[k]}. {it.choices[k]}" for k in order)
            out.append(
                BenchItem(
                    f"fidelity:{it.id}",
                    self.name,
                    f"{it.prompt}\n\n{opts}\n\n{_INSTRUCTION}",
                    [it.choices[k] for k in order],
                    None,
                    {
                        "topic": it.topic,
                        "expected": {
                            fw: new_letter[e.answer] if e.answer else None
                            for fw, e in it.expected.items()
                        },
                        "confidence": {fw: e.confidence for fw, e in it.expected.items()},
                        "canonical_letter": new_letter,
                    },
                )
            )
        return out

    def score(self, item: BenchItem, response_text: str) -> list[ScoreTuple]:
        valid = "ABC"[: len(item.choices or [])]
        pred = extract_letter(response_text, valid)
        expected: dict[str, str | None] = item.meta["expected"]
        conf: dict[str, str] = item.meta["confidence"]
        scores = []
        for fw in FRAMEWORKS:
            exp = expected[fw]
            metric = f"fidelity__{fw}"
            if pred is None:
                scores.append(ScoreTuple(metric, NAN, "exact", "unparseable letter"))
            elif exp is None:
                scores.append(
                    ScoreTuple(metric, NAN, "exact", f"pred={pred}; framework indeterminate")
                )
            else:
                scores.append(
                    ScoreTuple(
                        metric,
                        float(pred == exp),
                        "exact",
                        f"pred={pred} expected={exp} ({conf[fw]})",
                    )
                )
        scores.append(
            unparseable_score(pred is None, "fidelity__* are NaN" if pred is None else "")
        )
        return scores
