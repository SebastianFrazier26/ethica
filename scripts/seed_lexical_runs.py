"""Seed and check the D-58 lexical-control runs without touching the original runs/ dirs.

    uv run python scripts/seed_lexical_runs.py seed     # copy original DB + RAG indexes
    uv run python scripts/seed_lexical_runs.py pending  # what `ethica run` would still do

`seed` uses SQLite's online backup API, which reads a consistent snapshot including any WAL
content (abl-rag-replicate/ has a -wal file), instead of a file copy that could miss it. It
refuses to overwrite an existing target DB. `ethica run --dry-run` only sizes the grid; it does
not consult the store, so `pending` replays generate()/judge()'s resume logic read-only.
"""

from __future__ import annotations

import math
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from ethica.config import load_config, resolve_model, resolve_scenarios
from ethica.orchestrator import _needs_run, _tasks_for, expand_grid, load_bench_items
from ethica.scoring.judge import BENCHMARK_RUBRICS, applicable_rubrics
from ethica.tracking.store import Store

ROOT = Path(__file__).resolve().parent.parent
PAIRS = {
    "configs/experiments/abl-rag-lexical.yaml": "runs/abl-rag-frameworks",
    "configs/experiments/abl-rag-lexical-replicate.yaml": "runs/abl-rag-replicate",
}


def seed() -> None:
    for cfg_path, src in PAIRS.items():
        cfg = load_config(ROOT / cfg_path)
        src_dir, dst_dir = ROOT / src, ROOT / cfg.resolved_out_dir
        dst_db = dst_dir / "ethica.db"
        if dst_db.exists():
            print(f"skip {dst_db.relative_to(ROOT)}: exists")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        s = sqlite3.connect(f"file:{src_dir / 'ethica.db'}?mode=ro", uri=True)
        d = sqlite3.connect(dst_db)
        with d:
            s.backup(d)
        s.close()
        d.close()
        if (src_dir / "rag").is_dir():
            shutil.copytree(src_dir / "rag", dst_dir / "rag", dirs_exist_ok=True)
        print(f"seeded {dst_db.relative_to(ROOT)} from {src}/ethica.db")


def pending() -> None:
    for cfg_path in PAIRS:
        cfg = load_config(ROOT / cfg_path)
        db = ROOT / cfg.resolved_out_dir / "ethica.db"
        if not db.exists():
            # never open the original DB through Store: closing it checkpoints/removes its WAL
            sys.exit(f"{cfg_path}: {db.relative_to(ROOT)} missing; run `seed` first")
        store = Store(f"sqlite:///{db}")
        try:
            status = store.episode_status(cfg.name)
            bench = load_bench_items(cfg)
            gen: Counter[str] = Counter()
            reuse: Counter[str] = Counter()
            new_judge: Counter[str] = Counter()
            for v in expand_grid(cfg):
                for t in _tasks_for(cfg, v, [], bench):
                    run, _ = _needs_run(status.get(t.episode_id))
                    (gen if run else reuse)[v.framework.name] += 1
                    if run and t.item is not None and t.item.benchmark in BENCHMARK_RUBRICS:
                        new_judge[v.framework.name] += 1

            scorer = f"judge:{resolve_model(cfg.judge.model, cfg).key}"
            done: dict[str, set[str]] = {}
            prior: dict[str, dict[str, float]] = {}
            for r in store.scores_table(cfg.name):
                if r["scorer"] == scorer:
                    done.setdefault(r["episode_id"], set()).add(r["metric"])
                elif not r["scorer"].startswith("judge:") and not math.isnan(r["value"]):
                    prior.setdefault(r["episode_id"], {})[r["metric"]] = r["value"]
            scen = {s.id: s for s in resolve_scenarios(cfg)}
            old_judge = 0
            for ep in store.iter_episodes(cfg.name, include_errors=False):
                _, rubrics = applicable_rubrics(ep, scen, prior_scores=prior)
                if rubrics and not all(r.metric in done.get(ep.episode_id, ()) for r in rubrics):
                    old_judge += 1
        finally:
            store.close()
        print(f"\n{cfg_path} (experiment {cfg.name}, db {db.relative_to(ROOT)})")
        for fw in sorted(set(gen) | set(reuse)):
            print(
                f"  {fw:18s} generate {gen[fw]:5d}  reuse {reuse[fw]:5d}  judge {new_judge[fw]:4d}"
            )
        print(
            f"  total: generate {sum(gen.values())}, reuse {sum(reuse.values())}, "
            f"judge {sum(new_judge.values())} new + {old_judge} stored episodes not yet judged"
        )


if __name__ == "__main__":
    {"seed": seed, "pending": pending}[sys.argv[1] if len(sys.argv) > 1 else "pending"]()
