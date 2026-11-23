"""Lexical-confound check for the v0 RAG result (D-58, Part A). No model needed.

BM25 retrieval is deterministic, so the 1000-token block each harmful prompt received is
recomputed with the harness's own code (RAGConditioner + render_block; episode rows do not
store it). Per block: harm-lexicon density and overlap with the request's own content words
(ethica.analysis.lexical). Outputs, tracked, in docs/study-v0/data/:

- lexical_contexts.csv      framework x task densities (ids and numbers only, no text)
- lexical_by_framework.csv  per model x framework means over that model's harmful tasks
- lexical_correlation.csv   Spearman: framework density (minus placebo) vs risk difference
- lexical_task_level.csv    qwen3-4b-abl episode-level logistic fits of the judge verdict
- lexical_binned.csv        qwen3-4b-abl verdict rate by within-framework density tertile

Run: uv run python scripts/lexical_analysis.py
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ethica.analysis.lexical import Density, density, logistic_fit, spearman
from ethica.benchmarks import registry as bench_registry
from ethica.conditioning.corpus import corpus_built, load_corpus
from ethica.conditioning.prompt import render_block
from ethica.conditioning.rag import RAGConditioner, build_index
from ethica.config import load_config
from ethica.orchestrator import load_bench_items
from ethica.types import Message

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "study-v0" / "data"
CONFIGS = [
    ROOT / "configs/experiments/abl-rag-frameworks.yaml",
    ROOT / "configs/experiments/abl-rag-replicate.yaml",
]
DB_4B = ROOT / "runs/abl-rag-frameworks/ethica.db"
HARMFUL = ("harmbench", "xstest_unsafe")
ETHICS = [
    "virtue", "deontology", "care", "contractualism", "divine_command", "rule_utilitarian",
    "act_utilitarian",
]  # fmt: skip
EXTRA = ["placebo_harmvocab"]  # analysed when built; not in the v0 runs
JUDGE = "judge:judge-default"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def harmful_tasks() -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    """task_id -> (benchmark, prompt); model -> task ids it saw (any seed)."""
    prompts: dict[str, tuple[str, str]] = {}
    per_model: dict[str, set[str]] = defaultdict(set)
    for path in CONFIGS:
        cfg = load_config(path)
        for (name, _), items in load_bench_items(cfg).items():
            if name not in HARMFUL:
                continue
            for it in items:
                prompts[it.id] = (name, it.prompt)
                for m in cfg.models:
                    per_model[m].add(it.id)
    return prompts, per_model


def block_densities(
    fw_name: str, prompts: dict[str, tuple[str, str]], k: int, budget: int
) -> dict[str, tuple[Density, Density]]:
    """task -> (excerpts-only, whole block). Excerpt counts are block minus header, so the
    excerpt truncation render_block applies is respected exactly."""
    corpus = load_corpus(ROOT / "corpora" / fw_name)
    framework = corpus.framework
    cond = RAGConditioner(framework, build_index(corpus), k, budget)
    header = render_block(framework, [], budget)
    out: dict[str, tuple[Density, Density]] = {}
    for tid, (bench, prompt) in prompts.items():
        system = bench_registry.get(bench).system_prompt
        msgs = ([Message("system", system)] if system else []) + [Message("user", prompt)]
        block = render_block(framework, cond.retrieve(msgs), budget)
        whole, head = density([block], prompt), density([header], prompt)
        excerpts = Density(
            whole.n_tokens - head.n_tokens,
            whole.harm_hits - head.harm_hits,
            whole.overlap_hits - head.overlap_hits,
        )
        out[tid] = (excerpts, whole)
    return out


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else math.nan


def task_level(dens: dict[str, dict[str, tuple[Density, Density]]]) -> None:
    con = sqlite3.connect(f"file:{DB_4B}?mode=ro", uri=True)
    rows = con.execute(
        """select v.framework, e.task_id, e.seed, s.value
           from scores s join episodes e on e.episode_id = s.episode_id
           join variants v on v.variant_id = e.variant_id
           where s.metric = 'harmful_compliance' and s.scorer = ? and s.value is not null
             and e.error is null""",
        (JUDGE,),
    ).fetchall()
    con.close()
    fws = ["placebo", *ETHICS]
    obs = [
        (fw, tid, float(val), dens[fw][tid][0].harm, dens[fw][tid][0].overlap)
        for fw, tid, _, val in rows
        if fw in fws and tid in dens.get(fw, {}) and not math.isnan(val)
    ]
    y = np.array([o[2] for o in obs])
    harm = np.array([o[3] for o in obs])
    over = np.array([o[4] for o in obs])

    def z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / v.std()

    fw_dummies = np.array([[o[0] == f for f in fws[1:]] for o in obs], dtype=np.float64)
    tasks = sorted({o[1] for o in obs})
    task_dummies = np.array([[o[1] == t for t in tasks[1:]] for o in obs], dtype=np.float64)
    ones = np.ones((len(obs), 1))
    out: list[dict[str, Any]] = []
    specs = {
        # framework fixed effects: density slope from across-task variation within framework
        "framework_FE": (fw_dummies, 1e-6),
        # task fixed effects: slope from across-framework variation within a prompt, i.e. the
        # lexical reading itself; ridge 0.1 tames tasks that are all-0 / all-1
        "task_FE": (task_dummies, 0.1),
    }
    for spec, (fe, ridge) in specs.items():
        for label, cols in (
            ("harm", [z(harm)]),
            ("overlap", [z(over)]),
            ("harm+overlap", [z(harm), z(over)]),
        ):
            x = np.column_stack([ones, *cols, fe])
            fit = logistic_fit(x, y, ridge=ridge)
            names = label.split("+")
            for j, name in enumerate(names, start=1):
                out.append(
                    {
                        "model": "qwen3-4b-abl",
                        "spec": spec,
                        "predictors": label,
                        "term": f"z_{name}_density",
                        "coef_logodds_per_sd": float(fit.coef[j]),
                        "se_naive": float(fit.se[j]),
                        "z": float(fit.coef[j] / fit.se[j]),
                        "n_episodes": fit.n,
                        "converged": fit.converged,
                    }
                )
    write_csv(DATA / "lexical_task_level.csv", out)

    binned: list[dict[str, Any]] = []
    for fw in fws:
        sub = [(o[3], o[2]) for o in obs if o[0] == fw]
        cuts = np.quantile([d for d, _ in sub], [1 / 3, 2 / 3])
        for b, name in enumerate(("low", "mid", "high")):
            sel = [
                v
                for d, v in sub
                if (b == 0 and d <= cuts[0])
                or (b == 1 and cuts[0] < d <= cuts[1])
                or (b == 2 and d > cuts[1])
            ]
            binned.append(
                {
                    "framework": fw,
                    "harm_density_tertile": name,
                    "n_episodes": len(sel),
                    "harmful_compliance_rate": mean(sel),
                }
            )
    write_csv(DATA / "lexical_binned.csv", binned)


def main() -> None:
    cfg = load_config(CONFIGS[0])
    k, budget = cfg.rag_k, cfg.budget_tokens
    for path in CONFIGS[1:]:
        other = load_config(path)
        assert (other.rag_k, other.budget_tokens) == (k, budget), "configs differ in k/budget"
    prompts, per_model = harmful_tasks()
    fws = ["placebo", *ETHICS, *(f for f in EXTRA if corpus_built(ROOT / "corpora" / f))]
    dens = {fw: block_densities(fw, prompts, k, budget) for fw in fws}

    write_csv(
        DATA / "lexical_contexts.csv",
        [
            {
                "framework": fw,
                "task_id": tid,
                "benchmark": prompts[tid][0],
                "excerpt_tokens": ex.n_tokens,
                "harm_density": ex.harm,
                "overlap_density": ex.overlap,
                "block_harm_density": whole.harm,
            }
            for fw in fws
            for tid, (ex, whole) in sorted(dens[fw].items())
        ],
    )

    by_fw: list[dict[str, Any]] = []
    means: dict[tuple[str, str], tuple[float, float]] = {}
    for model, tids in per_model.items():
        for fw in fws:
            ex = [dens[fw][t][0] for t in tids]
            hb = [dens[fw][t][0].harm for t in tids if prompts[t][0] == "harmbench"]
            xs = [dens[fw][t][0].harm for t in tids if prompts[t][0] == "xstest_unsafe"]
            h, o = mean([d.harm for d in ex]), mean([d.overlap for d in ex])
            means[(model, fw)] = (h, o)
            by_fw.append(
                {
                    "model": model,
                    "framework": fw,
                    "n_tasks": len(ex),
                    "harm_density": h,
                    "harm_density_harmbench": mean(hb),
                    "harm_density_xstest_unsafe": mean(xs),
                    "overlap_density": o,
                    "block_harm_density": mean([dens[fw][t][1].harm for t in tids]),
                }
            )
    write_csv(DATA / "lexical_by_framework.csv", by_fw)

    with (DATA / "contrasts_vs_placebo.csv").open(encoding="utf-8") as fh:
        rd = {(r["model"], r["framework"]): float(r["risk_diff"]) for r in csv.DictReader(fh)}
    corr: list[dict[str, Any]] = []
    for scope in ["pooled", *per_model]:
        keys = [k for k in rd if scope in ("pooled", k[0]) and k[1] in ETHICS]
        for j, var in enumerate(("harm_density", "overlap_density")):
            x = [means[k][j] - means[(k[0], "placebo")][j] for k in keys]
            s = spearman(x, [rd[k] for k in keys])
            corr.append(
                {
                    "scope": scope,
                    "x": f"{var}_minus_placebo",
                    "y": "risk_diff_vs_placebo",
                    "rho": s.rho,
                    "p_perm": s.p,
                    "n": s.n,
                    "exact": s.exact,
                }
            )
    write_csv(DATA / "lexical_correlation.csv", corr)
    task_level(dens)


if __name__ == "__main__":
    main()
