"""Validate the judge against human labels: blind export, label import, kappa."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ethica.scenarios.schema import Scenario
from ethica.scoring.judge import Judge, Rubric, rubric_for_metric
from ethica.types import EpisodeResult, Score

LABELS = ("yes", "no", "unclear")
_ALIASES = {
    "yes": "yes",
    "y": "yes",
    "1": "yes",
    "true": "yes",
    "no": "no",
    "n": "no",
    "0": "no",
    "false": "no",
    "unclear": "unclear",
    "u": "unclear",
    "?": "unclear",
}


def item_id(episode_id: str, metric: str) -> str:
    return f"{episode_id}::{metric}"


def verdict_of(value: float) -> str:
    if math.isnan(value):
        return "unclear"
    return "yes" if value >= 0.5 else "no"


def _key_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.key{path.suffix}")


def _write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            w.writerows(rows)
    else:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )


def _read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def export_label_sample(
    judge: Judge,
    episodes: Sequence[EpisodeResult],
    scores: Sequence[Score],
    n: int,
    seed: int,
    path: Path,
    *,
    metric: str,
    scenario_lookup: Mapping[str, Scenario] | None = None,
) -> int:
    """Write a blind labelling sheet to `path` (.jsonl or .csv) and the judge's verdicts to
    `<stem>.key<suffix>`. The sheet renders each episode exactly as the judge saw it, so
    humans and judge face the same view. Returns the number of rows written.
    """
    rubric: Rubric = rubric_for_metric(metric)
    by_id = {e.episode_id: e for e in episodes}
    pool = sorted(
        (s for s in scores if s.metric == metric and s.scorer.startswith("judge:")),
        key=lambda s: s.episode_id,
    )
    pool = [s for s in pool if s.episode_id in by_id]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False).tolist())

    sheet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for i in idx:
        s = pool[i]
        ep = by_id[s.episode_id]
        scenario = (scenario_lookup or {}).get(ep.task_id)
        rendered = judge.render(ep, rubric, scenario)
        iid = item_id(ep.episode_id, metric)
        sheet.append(
            {
                "item_id": iid,
                "rubric": rubric.id,
                "question": rubric.question,
                "transcript": rendered.text,
                "label": "",
            }
        )
        key.append(
            {
                "item_id": iid,
                "judge_verdict": verdict_of(s.value),
                "judge_rationale": s.rationale,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, sheet, ["item_id", "rubric", "question", "transcript", "label"])
    _write(_key_path(path), key, ["item_id", "judge_verdict", "judge_rationale"])
    return len(sheet)


def import_labels(path: Path) -> dict[str, str]:
    """Rows with an empty label are skipped; unknown label strings raise ValueError."""
    out: dict[str, str] = {}
    for row in _read(path):
        raw = str(row.get("label", "")).strip().lower()
        if not raw:
            continue
        if raw not in _ALIASES:
            raise ValueError(f"unrecognised label {raw!r} for item {row.get('item_id')}")
        out[str(row["item_id"])] = _ALIASES[raw]
    return out


def import_key(path: Path) -> dict[str, str]:
    return {str(r["item_id"]): str(r["judge_verdict"]) for r in _read(path)}


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return math.nan
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted Cohen's kappa. NaN when undefined (no items, or chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return math.nan
    cats = sorted(set(a) | set(b))
    pos = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b, strict=True):
        table[pos[x], pos[y]] += 1
    n = table.sum()
    p_o = np.trace(table) / n
    p_e = float(table.sum(axis=1) @ table.sum(axis=0)) / (n * n)
    if math.isclose(p_e, 1.0):
        return math.nan
    return float((p_o - p_e) / (1.0 - p_e))


def agreement_report(
    human: Mapping[str, str], judge_key: Mapping[str, str], *, drop_unclear: bool = True
) -> dict[str, float]:
    """Pair labels by item_id. With drop_unclear, items either side marked unclear are
    excluded from kappa/agreement and counted separately."""
    ids = sorted(set(human) & set(judge_key))
    a = [human[i] for i in ids]
    b = [judge_key[i] for i in ids]
    n_unclear = sum(1 for x, y in zip(a, b, strict=True) if "unclear" in (x, y))
    if drop_unclear:
        kept = [(x, y) for x, y in zip(a, b, strict=True) if "unclear" not in (x, y)]
        a = [x for x, _ in kept]
        b = [y for _, y in kept]
    return {
        "n_paired": float(len(ids)),
        "n_scored": float(len(a)),
        "n_unclear": float(n_unclear),
        "percent_agreement": percent_agreement(a, b),
        "kappa": cohens_kappa(a, b),
    }
