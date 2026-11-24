import re
from pathlib import Path

import pytest
import yaml

from ethica.scoring.judge import FRAMEWORK_KEY_TERMS

ROOT = Path(__file__).resolve().parent.parent
PLAN = yaml.safe_load((ROOT / "configs/training/v1-pilot.yaml").read_text())
ETHICS = [a for a in PLAN["arms"] if a["kind"] == "ethics"]


@pytest.mark.parametrize("arm", ETHICS, ids=lambda a: a["name"])
def test_signed_off_commitments_exist_and_are_short(arm):
    text = (ROOT / arm["commitments"]).read_text(encoding="utf-8")
    points = re.findall(r"^\d+\. ", text, re.MULTILINE)
    assert 5 <= len(points) <= 7  # owner, 2026-11-23: about six points each


@pytest.mark.parametrize("arm", ETHICS, ids=lambda a: a["name"])
def test_commitments_avoid_the_terms_the_screen_rejects(arm):
    # the teacher echoes its system prompt; a term here would get its answers screened as "named"
    text = (ROOT / arm["commitments"]).read_text(encoding="utf-8")
    hits = [
        t
        for t in FRAMEWORK_KEY_TERMS[arm["name"]]
        if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)
    ]
    assert hits == []
