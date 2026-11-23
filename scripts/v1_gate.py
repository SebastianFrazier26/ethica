"""Internalisation gate report for one experiment (docs/study-v1/GATE.md).

    uv run python scripts/v1_gate.py configs/experiments/v1-gate-generic.yaml

Reads the experiment's stored scores (no model is loaded) and writes gate.md, gate.csv,
confusion.csv and persistence.csv under <out_dir>/gate/. The experiment must include the
`open_dilemmas` benchmark with the judge enabled; `fidelity` adds the fidelity columns and two
or more sampling seeds add order consistency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ethica.analysis.gate import confusion, gate_markdown, gate_rows, persistence
from ethica.config import load_config
from ethica.orchestrator import expand_grid
from ethica.tracking.store import Store


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = Store(cfg.resolved_db_url)
    try:
        rows = store.scores_table(
            cfg.name, include_errors=False, variant_ids={v.variant_id for v in expand_grid(cfg)}
        )
    finally:
        store.close()
    gate, conf = gate_rows(rows), confusion(rows)
    out = cfg.resolved_out_dir / "gate"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "gate.csv", [asdict(g) for g in gate])
    labels = sorted({lab for shares in conf.values() for lab in shares})
    _write(
        out / "confusion.csv",
        [{"arm": a, **{lab: f"{s.get(lab, 0.0):.3f}" for lab in labels}} for a, s in conf.items()],
    )
    _write(out / "persistence.csv", persistence(gate))
    md = gate_markdown(gate, conf)
    (out / "gate.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
