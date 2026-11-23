"""Harmful-compliance contrasts against a second baseline, `placebo_harmvocab` (D-58, Part C).

The report (analysis/report.py) contrasts only `kind: ethics` arms against one baseline, so it
cannot give placebo vs placebo_harmvocab, and running it with baseline=placebo_harmvocab would
drop the primary placebo contrast. This reproduces the report's contrast exactly (judge
scorer, matched task x seed pairs, task-cluster bootstrap CI, task-level sign-flip permutation,
Holm across ethics frameworks within model) for every arm against any reference.

    uv run python scripts/lexical_contrasts.py configs/experiments/abl-rag-lexical.yaml
    ... --reference placebo     # sanity check: must equal the report's vs-placebo numbers

Writes <out_dir>/report/contrasts_vs_<reference>.csv and prints the table.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ethica.analysis.stats import (
    holm_bonferroni,
    mean_ci_cluster_bootstrap,
    paired_permutation_test,
)
from ethica.config import load_config
from ethica.orchestrator import expand_grid
from ethica.tracking.store import Store

METRIC = "harmful_compliance"
Cell = dict[tuple[str, int], float]


def cells(cfg_path: Path) -> tuple[dict[tuple[str, str], Cell], dict[str, str], Path, Any]:
    cfg = load_config(cfg_path)
    grid = {v.variant_id for v in expand_grid(cfg)}
    store = Store(cfg.resolved_db_url)
    try:
        rows = store.scores_table(cfg.name, [METRIC], variant_ids=grid)
    finally:
        store.close()
    acc: dict[tuple[str, str], dict[tuple[str, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kinds: dict[str, str] = {}
    for r in rows:
        if not r["scorer"].startswith("judge:") or math.isnan(r["value"]):
            continue
        kinds[r["framework"]] = r["framework_kind"]
        acc[(r["model_key"], r["framework"])][(r["task_id"], r["seed"])].append(r["value"])
    out = {k: {kk: float(np.mean(v)) for kk, v in c.items()} for k, c in acc.items()}
    return out, kinds, cfg.resolved_out_dir, cfg


def contrast(a: Cell, b: Cell, n_boot: int, n_perm: int, seed: int) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    av, bv = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    tasks = [k[0] for k in keys]
    ci = mean_ci_cluster_bootstrap(av - bv, tasks, n_boot=n_boot, seed=seed)
    perm = paired_permutation_test(av, bv, keys, n_perm=n_perm, seed=seed, clusters=tasks)
    return {
        "rate": float(av.mean()),
        "rate_ref": float(bv.mean()),
        "risk_diff": float(av.mean() - bv.mean()),
        "rd_lo": ci.lo,
        "rd_hi": ci.hi,
        "p_perm": perm.p,
        "n_pairs": len(keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--reference", default="placebo_harmvocab")
    args = ap.parse_args()
    data, kinds, out_dir, cfg = cells(args.config)
    r = cfg.report
    rows: list[dict[str, Any]] = []
    for model in cfg.models:
        if (model, args.reference) not in data:
            print(f"{model}: no judged {args.reference} episodes yet")
            continue
        ref = data[(model, args.reference)]
        batch = []
        for fw in sorted(f for m, f in data if m == model and f not in (args.reference, "none")):
            c = contrast(data[(model, fw)], ref, r.n_boot, r.n_perm, 0)
            batch.append({"model": model, "framework": fw, "kind": kinds[fw], **c})
        ethics = [b for b in batch if b["kind"] == "ethics"]
        for b, p in zip(ethics, holm_bonferroni([b["p_perm"] for b in ethics]), strict=True):
            b["p_holm"] = float(p)
        for b in batch:
            b.setdefault("p_holm", math.nan)
        rows.extend(sorted(batch, key=lambda b: b["risk_diff"]))
    if not rows:
        return
    path = out_dir / "report" / f"contrasts_vs_{args.reference}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'model':20s} {'framework':18s} {'rd':>7s} {'95% CI':>17s} {'p_holm':>7s} {'n':>4s}")
    for b in rows:
        print(
            f"{b['model']:20s} {b['framework']:18s} {b['risk_diff']:7.3f} "
            f"[{b['rd_lo']:6.3f}, {b['rd_hi']:6.3f}] {b['p_holm']:7.4f} {b['n_pairs']:4d}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
