from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from judge_stubs import StubBackend, make_episode, make_scenario

from ethica.scoring.agreement import (
    agreement_report,
    cohens_kappa,
    export_label_sample,
    import_key,
    import_labels,
    percent_agreement,
)
from ethica.scoring.judge import Judge
from ethica.types import Score


def test_kappa_hand_computed() -> None:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: po=.7, pe=.5*.6+.5*.4=.5, kappa=.4
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert percent_agreement(a, b) == pytest.approx(0.7)
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_edges() -> None:
    assert cohens_kappa(["yes", "no"], ["yes", "no"]) == pytest.approx(1.0)
    assert cohens_kappa(["yes", "no"], ["no", "yes"]) == pytest.approx(-1.0)
    assert math.isnan(cohens_kappa(["yes"] * 3, ["yes"] * 3))
    assert math.isnan(cohens_kappa([], []))
    with pytest.raises(ValueError):
        cohens_kappa(["yes"], [])


@pytest.mark.parametrize("suffix", [".jsonl", ".csv"])
def test_export_blind_and_roundtrip(tmp_path: Path, suffix: str) -> None:
    eps = [make_episode(episode_id=f"e{i}") for i in range(6)]
    scores = [
        Score(f"e{i}", "coercion", float(i % 2) if i != 5 else math.nan, "judge:m", "why")
        for i in range(6)
    ]
    scores.append(Score("e0", "coercion", 1.0, "programmatic:x"))
    path = tmp_path / f"sheet{suffix}"
    judge = Judge(StubBackend([]), "m")
    n = export_label_sample(
        judge, eps, scores, 4, 7, path, metric="coercion", scenario_lookup={"s1": make_scenario()}
    )
    assert n == 4
    sheet = path.read_text()
    assert "why" not in sheet and "judge_verdict" not in sheet
    key_path = tmp_path / f"sheet.key{suffix}"
    assert key_path.exists()

    again = tmp_path / f"again{suffix}"
    export_label_sample(judge, eps, scores, 4, 7, again, metric="coercion")
    assert [r for r in import_key(key_path)] == [
        r for r in import_key(tmp_path / f"again.key{suffix}")
    ]

    key = import_key(key_path)
    # a human who agrees with the judge on everything
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in sheet.splitlines()]
        for r in rows:
            r["label"] = key[r["item_id"]]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        import csv

        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["label"] = key[r["item_id"]]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    labels = import_labels(path)
    assert labels == key
    rep = agreement_report(labels, key)
    assert rep["percent_agreement"] == 1.0 and rep["n_paired"] == 4.0


def test_import_labels_rejects_garbage(tmp_path: Path) -> None:
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"item_id": "a", "label": "maybe"}) + "\n")
    with pytest.raises(ValueError):
        import_labels(p)
    p.write_text(
        json.dumps({"item_id": "a", "label": "Y"})
        + "\n"
        + json.dumps({"item_id": "b", "label": ""})
        + "\n"
    )
    assert import_labels(p) == {"a": "yes"}


def test_agreement_report_drops_unclear() -> None:
    human = {"a": "yes", "b": "no", "c": "unclear", "d": "yes"}
    judge = {"a": "yes", "b": "yes", "c": "no", "d": "unclear"}
    rep = agreement_report(human, judge)
    assert rep["n_unclear"] == 2.0 and rep["n_scored"] == 2.0
    assert rep["percent_agreement"] == 0.5
