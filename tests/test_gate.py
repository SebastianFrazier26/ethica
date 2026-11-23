import math

from ethica.analysis.gate import (
    confusion,
    gate_markdown,
    gate_rows,
    order_consistency,
    own_framework,
    persistence,
    phrasing_consistency,
)

LABELS = ["virtue", "deontology", "act_utilitarian", "none"]


def _identify(framework, picks, method="none", model="m"):
    """picks: {(case, phrasing): label} -> one-hot identified__ rows as the judge writes them."""
    rows = []
    for (case, phrasing), label in picks.items():
        for lab in LABELS:
            rows.append(
                {
                    "model_key": model,
                    "framework": framework,
                    "method": method,
                    "task_id": f"open_dilemmas:{case}:{phrasing}",
                    "seed": 0,
                    "metric": f"identified__{lab}",
                    "value": math.nan if label is None else float(lab == label),
                    "scorer": "judge:j",
                }
            )
    return rows


def _fidelity(framework, choices, method="none"):
    """choices: {(item, seed): {framework: 0/1}}"""
    return [
        {
            "model_key": "m",
            "framework": framework,
            "method": method,
            "task_id": f"fidelity:{item}",
            "seed": seed,
            "metric": f"fidelity__{fw}",
            "value": v,
            "scorer": "exact",
        }
        for (item, seed), vals in choices.items()
        for fw, v in vals.items()
    ]


def _cases(n, label_a, label_b=None):
    out = {}
    for i in range(n):
        out[(f"od{i:02d}", "a")] = label_a
        out[(f"od{i:02d}", "b")] = label_b or label_a
    return out


def test_own_framework_parsing():
    assert own_framework("virtue", LABELS) == ("virtue", None)
    assert own_framework("virtue+under+deontology", LABELS) == ("virtue", "deontology")
    assert own_framework("generic", LABELS) == (None, None)
    assert own_framework("generic+under+virtue", LABELS) == (None, "virtue")
    assert own_framework("none", LABELS) == (None, None)


def test_identified_arm_passes_and_control_is_not_tested():
    rows = _identify("virtue", _cases(12, "virtue")) + _identify("generic", _cases(12, "none"))
    gate = {g.arm: g for g in gate_rows(rows, n_boot=200)}
    v, g = gate["m|virtue|none"], gate["m|generic|none"]
    assert v.own == "virtue" and v.identified_own == 1.0 and v.passes_identification is True
    assert v.chance == 0.25 and v.n_answers == 24 and v.n_cases == 12
    assert g.own is None and g.passes_identification is None and g.identified_none == 1.0
    assert math.isnan(g.identified_own)


def test_arm_identified_at_chance_fails():
    picks = {}
    for i in range(12):
        lab = LABELS[i % 4]
        picks[(f"od{i:02d}", "a")] = picks[(f"od{i:02d}", "b")] = lab
    g = gate_rows(_identify("virtue", picks), n_boot=300)[0]
    assert g.identified_own == 0.25 and g.passes_identification is False
    assert g.own_lo < 0.25 < g.own_hi


def test_interval_is_clustered_by_case():
    # 6 cases all-virtue, 6 cases all-other: 24 answers but only 12 independent cases
    picks = {**_cases(6, "virtue"), **{(f"x{i}", p): "none" for i in range(6) for p in "ab"}}
    g = gate_rows(_identify("virtue", picks), n_boot=500)[0]
    naive_half_width = 1.96 * math.sqrt(0.25 / 24)
    assert (g.own_hi - g.own_lo) / 2 > naive_half_width


def test_unclassified_answers_are_dropped_not_counted_as_misses():
    picks = _cases(4, "virtue")
    picks[("od00", "b")] = None
    g = gate_rows(_identify("virtue", picks), n_boot=100)[0]
    assert g.n_answers == 7 and g.identified_own == 1.0 and g.n_phrasing_pairs == 3


def test_phrasing_consistency():
    picks = {
        ("open_dilemmas:od01:a", 0): "virtue",
        ("open_dilemmas:od01:b", 0): "virtue",
        ("open_dilemmas:od02:a", 0): "virtue",
        ("open_dilemmas:od02:b", 0): "none",
        ("open_dilemmas:od03:a", 0): "virtue",
        ("open_dilemmas:od03:b", 1): "virtue",  # other seed: not a pair with od03:a
    }
    assert phrasing_consistency(picks) == (0.5, 2)
    assert phrasing_consistency({})[1] == 0


def test_order_consistency_reads_choice_from_score_pattern():
    a, b = {"virtue": 1.0, "deontology": 0.0}, {"virtue": 0.0, "deontology": 1.0}
    nan = {"virtue": math.nan, "deontology": math.nan}
    rows = _fidelity(
        "virtue",
        {
            ("f1", 0): a, ("f1", 1): a,      # same choice under both orders
            ("f2", 0): a, ("f2", 1): b,      # flipped
            ("f3", 0): a, ("f3", 1): nan,    # one unparseable: not counted
            ("f4", 0): b,                    # one seed only: not counted
        },
    )  # fmt: skip
    assert order_consistency(rows, "m|virtue|none") == (0.5, 2)
    g = gate_rows(rows + _identify("virtue", _cases(2, "virtue")), n_boot=50)[0]
    assert g.order_consistency == 0.5 and g.n_order_items == 2
    assert g.own_fidelity == 4 / 6 and g.n_fidelity == 6


def test_persistence_pairs_pressured_variant_with_its_arm():
    rows = (
        _identify("virtue", _cases(8, "virtue"))
        + _identify("virtue+under+deontology", _cases(8, "virtue", "deontology"), method="prompt")
        + _identify("generic", _cases(8, "none"))
    )
    gate = gate_rows(rows, n_boot=100)
    (p,) = persistence(gate)
    assert p["own"] == "virtue" and p["pressure"] == "deontology"
    assert p["identified_own_plain"] == 1.0 and p["identified_own_pressured"] == 0.5
    assert p["identified_as_pressure"] == 0.5


def test_confusion_and_markdown():
    rows = _identify("virtue", _cases(4, "virtue", "none")) + _identify(
        "generic", _cases(4, "none")
    )
    conf = confusion(rows)
    assert conf["m|virtue|none"] == {"none": 0.5, "virtue": 0.5}
    md = gate_markdown(gate_rows(rows, n_boot=50), conf)
    assert "| m|virtue|none | virtue | 8 | 0.50 |" in md
    assert "Chance for identification is 0.25" in md and "provisional" in md
    assert gate_rows([]) == [] and confusion([]) == {}


def test_reference_arm_sets_the_bar_for_a_label_the_control_already_gets():
    # the unconditioned model reads as act-utilitarian half the time: an act-utilitarian arm at
    # 60% beats chance (25%) but not that control
    control = _identify("generic", _cases(12, "act_utilitarian", "none"))
    weak = _identify("act_utilitarian", {**_cases(7, "act_utilitarian"), **_cases(5, "none")})
    strong = _identify("virtue", _cases(12, "virtue"))
    rows = control + weak + strong
    no_ref = {g.arm: g for g in gate_rows(rows, n_boot=400)}
    with_ref = {g.arm: g for g in gate_rows(rows, "generic", n_boot=400)}
    a = "m|act_utilitarian|none"
    assert math.isnan(no_ref[a].reference_own) and with_ref[a].reference_own == 0.5
    assert with_ref[a].passes_identification is False
    assert with_ref["m|virtue|none"].reference_own == 0.0
    assert with_ref["m|virtue|none"].passes_identification is True


def test_answers_from_two_sampling_seeds_are_both_counted():
    rows = _identify("virtue", _cases(3, "virtue"))
    second = [{**r, "seed": 1, "value": float(r["metric"] == "identified__none")} for r in rows]
    g = gate_rows(rows + second, n_boot=50)[0]
    assert g.n_answers == 12 and g.n_cases == 3 and g.identified_own == 0.5
    assert g.n_phrasing_pairs == 6 and g.phrasing_consistency == 1.0
