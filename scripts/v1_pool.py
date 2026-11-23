"""Arm contrasts pooled over training seeds (study v1).

One experiment config per training seed (each evaluates that seed's adapters under the same
arm names):

    uv run python scripts/v1_pool.py --baseline generic \\
        configs/experiments/v1-pilot-s0.yaml configs/experiments/v1-pilot-s1.yaml ...

Reads stored scores only. Prints, per arm, the pooled difference from the baseline arm with an
interval that resamples training runs as well as tasks, and each run's own mean.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from ethica.analysis.pooling import Cell, cells_by_arm, pooled_contrast
from ethica.config import load_config
from ethica.orchestrator import expand_grid
from ethica.tracking.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", type=Path, nargs="+")
    ap.add_argument("--baseline", required=True, help="framework name of the baseline arm")
    ap.add_argument("--metric", default="harmful_compliance")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    runs: dict[str, dict[str, Cell]] = defaultdict(dict)  # arm -> run id -> cell
    for path in args.configs:
        cfg = load_config(path)
        store = Store(cfg.resolved_db_url)
        try:
            rows = store.scores_table(
                cfg.name,
                [args.metric],
                include_errors=False,
                variant_ids={v.variant_id for v in expand_grid(cfg)},
            )
        finally:
            store.close()
        for arm, cell in cells_by_arm(rows, args.metric).items():
            runs[arm][cfg.name] = cell

    out = []
    for arm in sorted(runs):
        model, framework, method = arm.split("|")
        ref = runs.get(f"{model}|{args.baseline}|{method}")
        if framework == args.baseline or ref is None:
            continue
        c = pooled_contrast(runs[arm], ref)
        out.append(
            {
                "arm": arm,
                "metric": args.metric,
                "diff_vs_baseline": f"{c.estimate:.3f}",
                "lo": f"{c.lo:.3f}",
                "hi": f"{c.hi:.3f}",
                "arm_runs": " ".join(f"{v:.2f}" for v in c.arm_run_means.values()),
                "baseline_runs": " ".join(f"{v:.2f}" for v in c.ref_run_means.values()),
                "n_tasks": c.n_tasks,
                "runs_resampled": c.runs_resampled,
                "sign_consistent": c.sign_consistent,
            }
        )
    if not out:
        raise SystemExit(f"no arm shares a model and method with baseline {args.baseline!r}")
    cols = list(out[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if not all(r["runs_resampled"] for r in out):
        print("\nNote: at least one arm has a single training run; its interval covers tasks only.")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
