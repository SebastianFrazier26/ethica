"""Markdown + CSV + PNG report for one experiment (DECISIONS D-12). See docs/ANALYSIS.md."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from ethica.analysis.stats import (
    benjamini_hochberg,
    cohens_h,
    holm_bonferroni,
    mean_ci_cluster_bootstrap,
    paired_permutation_test,
    wilson_ci,
)
from ethica.tracking.errors import is_context_overflow
from ethica.tracking.store import Store, scorer_family

PRIMARY_METRIC = "misaligned_action"
TAX_PREFIXES = ("capability_acc__", "ethics_acc__", "truthful_acc", "over_refusal", "task_success")
TAX_EXACT = {"harmful_compliance"}
# Metrics where the phrase heuristic is only an upper bound; the judge wins when it ran.
JUDGE_PREFERRED = frozenset({"over_refusal", "harmful_compliance"})
MAX_BAR_VARIANTS = 60

Cell = dict[tuple[str, int], float]


class VKey(NamedTuple):
    model_key: str
    safeguards: str
    framework: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.model_key}|{self.safeguards}|{self.framework}|{self.method}"


class Stratum(NamedTuple):
    model_key: str
    safeguards: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.model_key}|{self.safeguards}|{self.method}"


class ReportError(ValueError):
    """The stored data cannot be reported without silently mixing conditions."""


def _pick_scorers(
    rows: list[dict[str, Any]], prefer: str, overrides: Mapping[str, str]
) -> dict[str, str]:
    """One scorer *family* per metric so cells are comparable.

    Order: override > judge (JUDGE_PREFERRED metrics only) > `prefer` family > exact > other
    judges > heuristics. Selection is by family, not the raw scorer string, so a scenario whose
    check mix differs from the rest is not dropped. Ties go to the larger count, then name.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["metric"]][scorer_family(r["scorer"])] += 1

    def rank(metric: str, family: str) -> int:
        if family.startswith("heuristic:"):
            return 4
        if metric in JUDGE_PREFERRED and family.startswith("judge:"):
            return -1
        if family.startswith(prefer):
            return 0
        return 1 if family == "exact" else 2

    chosen = {}
    for metric, by_family in counts.items():
        override = scorer_family(overrides[metric]) if metric in overrides else None
        if override in by_family:
            chosen[metric] = override
        else:
            chosen[metric] = min(by_family, key=lambda f: (rank(metric, f), -by_family[f], f))
    return chosen


def _count(v: Any) -> int:
    if isinstance(v, list | tuple | dict | set):
        return len(v)
    return int(v) if isinstance(v, bool | int | float) else 0


def _arm_label(r: Mapping[str, Any]) -> str:
    return VKey(r["model_key"], r["safeguards"], r["framework"], r["method"]).label


def _exclusion_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per arm: how many episodes failed or were degraded, so exclusions are never silent."""
    acc: dict[str, dict[str, Any]] = {}
    for r in summaries:
        a = acc.setdefault(
            _arm_label(r),
            {
                "variant": _arm_label(r),
                "episodes": 0,
                "errored": 0,
                "context_overflow": 0,
                "retried": 0,
                "truncated": 0,
                "protocol_violation": 0,
                "unclosed_scratchpad": 0,
                "_blocks": [],
            },
        )
        meta = r["meta"]
        a["episodes"] += 1
        a["errored"] += r["error"] is not None
        a["context_overflow"] += is_context_overflow(r["error"])
        a["retried"] += int(meta.get("attempts", 1)) > 1
        a["truncated"] += bool(meta.get("truncated"))
        a["protocol_violation"] += _count(meta.get("protocol_violations")) > 0
        a["unclosed_scratchpad"] += _count(meta.get("unclosed_scratchpad")) > 0
        if "block_tokens_mean" in meta:
            a["_blocks"].append(float(meta["block_tokens_mean"]))
    out = []
    for a in sorted(acc.values(), key=lambda x: x["variant"]):
        blocks = a.pop("_blocks")
        a["mean_block_tokens"] = float(np.mean(blocks)) if blocks else math.nan
        out.append(a)
    return out


def _benchmark_rate_rows(
    summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per arm x benchmark: truncation and unparseable-answer rates. Arms that ramble hit the
    token cap or fail extraction more often, which would otherwise masquerade as a skill gap."""
    unparse = {r["episode_id"]: r["value"] for r in rows if r["metric"] == "benchmark_unparseable"}
    acc: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in summaries:
        if r["task_kind"] == "benchmark":
            acc[(_arm_label(r), str(r["meta"].get("benchmark", r["task_id"])))].append(r)
    out = []
    for (arm, bench), eps in sorted(acc.items()):
        scored = [unparse[e["episode_id"]] for e in eps if e["episode_id"] in unparse]
        out.append(
            {
                "variant": arm,
                "benchmark": bench,
                "n": len(eps),
                "truncated_rate": sum(bool(e["meta"].get("truncated")) for e in eps) / len(eps),
                "unparseable_rate": float(np.mean(scored)) if scored else math.nan,
            }
        )
    return out


def _heuristic_vs_judge(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_ep: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    arm_of: dict[tuple[str, str], str] = {}
    for r in rows:
        if r["metric"] not in JUDGE_PREFERRED:
            continue
        kind = (
            "heuristic"
            if r["family"].startswith("heuristic:")
            else "judge"
            if r["family"].startswith("judge:")
            else None
        )
        if kind is None:
            continue
        key = (r["metric"], r["episode_id"])
        by_ep[key][kind] = r["value"]
        arm_of[key] = _arm_label(r)
    acc: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for key, v in by_ep.items():
        if "heuristic" in v and "judge" in v:
            acc[(key[0], arm_of[key])].append((v["heuristic"], v["judge"]))
    return [
        {
            "metric": metric,
            "variant": arm,
            "n_both": len(pairs),
            "heuristic_rate": float(np.mean([h for h, _ in pairs])),
            "judge_rate": float(np.mean([j for _, j in pairs])),
            "agreement": float(np.mean([h == j for h, j in pairs])),
        }
        for (metric, arm), pairs in sorted(acc.items())
    ]


def _is_tax(metric: str) -> bool:
    return metric in TAX_EXACT or metric.startswith(TAX_PREFIXES)


def _fmt(x: float, nd: int = 3) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: list[dict[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append(
            "| "
            + " | ".join(_fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols)
            + " |"
        )
    return "\n".join(out) + "\n"


def _common(a: Cell, b: Cell) -> list[tuple[str, int]]:
    return sorted(set(a) & set(b))


def _task_means(cell: Cell) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for (task, _), v in cell.items():
        acc[task].append(v)
    return {t: float(np.mean(v)) for t, v in acc.items()}


def _is_binary(cells: Mapping[Any, Cell]) -> bool:
    return all(v in (0.0, 1.0) for c in cells.values() for v in c.values())


def _contrast(
    a: Cell,
    b: Cell,
    n_boot: int,
    n_perm: int,
    seed: int,
    permute_by: Literal["pair", "task"],
) -> dict[str, Any] | None:
    keys = _common(a, b)
    if not keys:
        return None
    av = np.array([a[k] for k in keys])
    bv = np.array([b[k] for k in keys])
    tasks = [k[0] for k in keys]
    ra, rb = float(av.mean()), float(bv.mean())
    ci = mean_ci_cluster_bootstrap(av - bv, tasks, n_boot=n_boot, seed=seed)
    perm = paired_permutation_test(
        av,
        bv,
        keys,
        n_perm=n_perm,
        seed=seed,
        clusters=tasks if permute_by == "task" else None,
    )
    in_unit = 0.0 <= ra <= 1.0 and 0.0 <= rb <= 1.0
    return {
        "n_pairs": len(keys),
        "n_tasks": len(set(tasks)),
        "rate": ra,
        "rate_ref": rb,
        "risk_diff": ra - rb,
        "rd_lo": ci.lo,
        "rd_hi": ci.hi,
        "cohens_h": cohens_h(ra, rb) if in_unit else math.nan,
        "p_perm": perm.p,
        "p_exact": perm.exact,
    }


def _contrast_rows(
    cm: Mapping[VKey, Cell],
    fw_keys: Mapping[str, VKey],
    arms: Sequence[str],
    ref: str,
    metric: str,
    st: Stratum,
    n_boot: int,
    n_perm: int,
    seed: int,
    permute_by: Literal["pair", "task"],
) -> list[dict[str, Any]]:
    rows = []
    for f in arms:
        c = _contrast(cm[fw_keys[f]], cm[fw_keys[ref]], n_boot, n_perm, seed, permute_by)
        if c is not None:
            rows.append(
                {
                    "metric": metric,
                    "stratum": st.label,
                    "model_key": st.model_key,
                    "safeguards": st.safeguards,
                    "method": st.method,
                    "framework": f,
                    "reference": ref,
                    **c,
                }
            )
    return rows


def build_report(
    store: Store,
    experiment_id: str,
    out_dir: str | Path,
    baseline: str = "placebo",
    extra_baselines: Sequence[str] = (),
    metrics: Sequence[str] | None = None,
    prefer_scorer: str = "programmatic",
    scorer_overrides: Mapping[str, str] | None = None,
    primary_metric: str = PRIMARY_METRIC,
    min_tasks: int = 10,
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int = 0,
    permute_by: Literal["pair", "task"] = "task",
    variant_ids: Collection[str] | None = None,
) -> Path:
    """`variant_ids` restricts the report to the current grid (orchestrator.expand_grid); older
    variants of the same experiment (changed config) are then never averaged in.

    `extra_baselines` are further reference arms (e.g. a second placebo). Each gets the same
    ethics-framework contrasts as `baseline`, Holm-corrected within itself, plus uncorrected
    contrasts of the other placebo arms against it. `baseline` stays the primary reference:
    the forest plots, exploratory BH and the interaction use it alone."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = tuple(dict.fromkeys(e for e in extra_baselines if e not in (baseline, "none")))
    counts = store.episode_counts(experiment_id, variant_ids)
    summaries = store.episode_summaries(experiment_id, variant_ids)
    all_rows = [
        {**r, "family": scorer_family(r["scorer"])}
        for r in store.scores_table(
            experiment_id, metrics, include_errors=True, variant_ids=variant_ids
        )
        if not math.isnan(r["value"])
    ]
    ids_per_arm: dict[VKey, set[str]] = defaultdict(set)
    for r in all_rows:
        ids_per_arm[VKey(r["model_key"], r["safeguards"], r["framework"], r["method"])].add(
            r["variant_id"]
        )
    ambiguous = {vk.label: sorted(v) for vk, v in ids_per_arm.items() if len(v) > 1}
    if ambiguous:
        raise ReportError(
            "several variant_ids map to one (model, safeguards, framework, method); pass the "
            f"current grid's variant_ids or rerun under a new experiment name: {ambiguous}"
        )
    chosen = _pick_scorers(all_rows, prefer_scorer, scorer_overrides or {})
    rows = [r for r in all_rows if chosen.get(r["metric"]) == r["family"]]

    vkind: dict[VKey, str] = {}
    vpair: dict[VKey, str] = {}
    cells: dict[str, dict[VKey, Cell]] = defaultdict(lambda: defaultdict(dict))
    dup: dict[tuple[str, VKey, str, int], list[float]] = defaultdict(list)
    for r in rows:
        vk = VKey(r["model_key"], r["safeguards"], r["framework"], r["method"])
        vkind[vk] = r["framework_kind"]
        vpair[vk] = r["pair_key"]
        dup[(r["metric"], vk, r["task_id"], r["seed"])].append(r["value"])
    for (metric, vk, task, sd), vals in dup.items():
        cells[metric][vk][(task, sd)] = float(np.mean(vals))

    metric_list = sorted(cells, key=lambda m: (m != primary_metric, m))
    n_analysed = len({r["episode_id"] for r in rows})
    n_tasks_all = len({r["task_id"] for r in rows})
    warnings: list[str] = []
    files: list[str] = []

    def save_fig(fig: Figure, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        files.append(name)

    # (a) rates
    rate_rows: dict[str, list[dict[str, Any]]] = {}
    for m in metric_list:
        binary = _is_binary(cells[m])
        rr = []
        for vk in sorted(cells[m]):
            cell = cells[m][vk]
            tasks = [k[0] for k in cell]
            iv = mean_ci_cluster_bootstrap(list(cell.values()), tasks, n_boot=n_boot, seed=seed)
            row: dict[str, Any] = {
                "variant": vk.label,
                "model_key": vk.model_key,
                "safeguards": vk.safeguards,
                "framework": vk.framework,
                "method": vk.method,
                "n": iv.n,
                "n_tasks": iv.n_clusters,
                "mean": iv.mean,
                "ci_lo": iv.lo,
                "ci_hi": iv.hi,
            }
            if binary:
                wl, wh = wilson_ci(sum(cell.values()), len(cell))
                row.update(wilson_lo=wl, wilson_hi=wh)
            if iv.n_clusters < min_tasks:
                warnings.append(f"`{m}` / `{vk.label}`: only {iv.n_clusters} tasks (< {min_tasks})")
            rr.append(row)
        rate_rows[m] = rr
        _write_csv(out / f"rates__{m}.csv", rr)
        files.append(f"rates__{m}.csv")
        shown = rr[:MAX_BAR_VARIANTS]
        fig = Figure(figsize=(7, max(2.0, 0.28 * len(shown) + 1)))
        ax = fig.subplots()
        y = np.arange(len(shown))
        mean = np.array([r["mean"] for r in shown])
        lo = np.array([r["ci_lo"] for r in shown])
        hi = np.array([r["ci_hi"] for r in shown])
        ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", capsize=2)
        ax.set_yticks(y, [r["variant"] for r in shown], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(f"{m} (mean, 95% cluster-bootstrap CI)")
        save_fig(fig, f"rates__{m}.png")

    # (b) contrasts, per stratum
    contrasts: dict[str, list[dict[str, Any]]] = {}
    for m in metric_list:
        cm = cells[m]
        strata = sorted({Stratum(k.model_key, k.safeguards, k.method) for k in cm})
        crow: list[dict[str, Any]] = []
        for st in strata:
            fw_keys = {
                k.framework: k for k in cm if Stratum(k.model_key, k.safeguards, k.method) == st
            }
            frameworks = sorted(f for f, k in fw_keys.items() if vkind[k] == "ethics")
            placebos = sorted(f for f, k in fw_keys.items() if vkind[k] == "placebo")

            args = (m, st, n_boot, n_perm, seed, permute_by)
            for ref in (baseline, "none", *extra):
                if ref not in fw_keys:
                    continue
                batch = _contrast_rows(cm, fw_keys, frameworks, ref, *args)
                adj = holm_bonferroni([b["p_perm"] for b in batch])
                for b, p in zip(batch, adj, strict=True):
                    b["p_holm"] = float(p)
                    if b["n_tasks"] < min_tasks:
                        warnings.append(
                            f"`{m}` contrast `{b['framework']}` vs `{ref}` in `{st.label}`: "
                            f"only {b['n_tasks']} tasks (< {min_tasks})"
                        )
                crow.extend(batch)
                if ref in extra:
                    # one placebo against another is a single descriptive comparison, kept
                    # out of the framework family so it does not change their Holm correction
                    others = [f for f in placebos if f != ref]
                    for b in _contrast_rows(cm, fw_keys, others, ref, *args):
                        b["p_holm"] = math.nan
                        crow.append(b)
        contrasts[m] = crow
    expl = [
        c
        for m in metric_list
        if m != primary_metric
        for c in contrasts[m]
        if c["reference"] == baseline
    ]
    bh = benjamini_hochberg([c["p_perm"] for c in expl])
    for c, p in zip(expl, bh, strict=True):
        c["p_bh_exploratory"] = float(p)
    for m in metric_list:
        for c in contrasts[m]:
            c.setdefault("p_bh_exploratory", math.nan)
        _write_csv(out / f"contrasts__{m}.csv", contrasts[m])
        if contrasts[m]:
            files.append(f"contrasts__{m}.csv")
            vs = [c for c in contrasts[m] if c["reference"] == baseline]
            if vs:
                fig = Figure(figsize=(7, max(2.0, 0.3 * len(vs) + 1)))
                ax = fig.subplots()
                y = np.arange(len(vs))
                rd = np.array([c["risk_diff"] for c in vs])
                lo = np.array([c["rd_lo"] for c in vs])
                hi = np.array([c["rd_hi"] for c in vs])
                ax.errorbar(rd, y, xerr=[rd - lo, hi - rd], fmt="o", capsize=2)
                ax.axvline(0, color="grey", lw=0.8)
                ax.set_yticks(y, [f"{c['framework']} @ {c['stratum']}" for c in vs], fontsize=6)
                ax.invert_yaxis()
                ax.set_xlabel(f"{m}: difference vs {baseline} (95% CI)")
                save_fig(fig, f"forest__{m}.png")

    # (c) safeguards interaction: stock and abliterated twins are paired through pair_key
    # (the abliterated model's base_of), since they are different model keys
    interaction: dict[str, list[dict[str, Any]]] = {}
    for m in metric_list:
        cm = cells[m]
        irows: list[dict[str, Any]] = []
        by_arm = {(vpair[k], k.safeguards, k.model_key, k.framework, k.method): k for k in cm}
        groups = sorted({(pk, method) for pk, _, _, _, method in by_arm})
        for pk, method in groups:
            stock_models = sorted(
                {mk for p, sg, mk, _, mt in by_arm if (p, sg, mt) == (pk, "stock", method)}
            )
            abl_models = sorted(
                {mk for p, sg, mk, _, mt in by_arm if (p, sg, mt) == (pk, "abliterated", method)}
            )
            for sm in stock_models:
                for am in abl_models:
                    for ref in (baseline, "none"):
                        batch = []
                        for f in sorted({k.framework for k in cm if vkind[k] == "ethics"}):
                            need = [
                                by_arm.get(key)
                                for key in (
                                    (pk, "stock", sm, f, method),
                                    (pk, "stock", sm, ref, method),
                                    (pk, "abliterated", am, f, method),
                                    (pk, "abliterated", am, ref, method),
                                )
                            ]
                            if any(k is None for k in need):
                                continue
                            f_s, r_s, f_a, r_a = (_task_means(cm[k]) for k in need if k is not None)
                            tasks = sorted(set(f_s) & set(r_s) & set(f_a) & set(r_a))
                            if not tasks:
                                continue
                            eff_s = np.array([f_s[t] - r_s[t] for t in tasks])
                            eff_a = np.array([f_a[t] - r_a[t] for t in tasks])
                            dd = eff_a - eff_s
                            ci = mean_ci_cluster_bootstrap(dd, tasks, n_boot=n_boot, seed=seed)
                            perm = paired_permutation_test(
                                dd, np.zeros_like(dd), tasks, n_perm=n_perm, seed=seed
                            )
                            batch.append(
                                {
                                    "metric": m,
                                    "model_key": f"{sm}/{am}",
                                    "method": method,
                                    "framework": f,
                                    "reference": ref,
                                    "n_tasks": len(tasks),
                                    "effect_stock": float(eff_s.mean()),
                                    "effect_abliterated": float(eff_a.mean()),
                                    "interaction": float(dd.mean()),
                                    "ci_lo": ci.lo,
                                    "ci_hi": ci.hi,
                                    "p_perm": perm.p,
                                }
                            )
                        adj = holm_bonferroni([b["p_perm"] for b in batch])
                        for b, p in zip(batch, adj, strict=True):
                            b["p_holm"] = float(p)
                        irows.extend(batch)
        interaction[m] = irows
        _write_csv(out / f"interaction__{m}.csv", irows)
        if irows:
            files.append(f"interaction__{m}.csv")

    # (d) fidelity manipulation check
    fid_metrics = [m for m in metric_list if m.startswith("fidelity__")]
    fid_rows: list[dict[str, Any]] = []
    for m in fid_metrics:
        for vk in sorted(cells[m]):
            tm = _task_means(cells[m][vk])
            fid_rows.append(
                {
                    "stratum": Stratum(vk.model_key, vk.safeguards, vk.method).label,
                    "framework": vk.framework,
                    "metric": m,
                    "n_items": len(tm),
                    "mean": float(np.mean(list(tm.values()))),
                }
            )
    _write_csv(out / "fidelity.csv", fid_rows)
    if fid_rows:
        files.append("fidelity.csv")
        strata_f = sorted({r["stratum"] for r in fid_rows})
        ncol = min(3, len(strata_f))
        nrow = math.ceil(len(strata_f) / ncol)
        fig = Figure(figsize=(4.5 * ncol, 3.5 * nrow))
        axes = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
        fw_all = sorted({r["framework"] for r in fid_rows})
        for ax, st in zip(axes, strata_f, strict=False):
            mat = np.full((len(fw_all), len(fid_metrics)), np.nan)
            for r in fid_rows:
                if r["stratum"] == st:
                    mat[fw_all.index(r["framework"]), fid_metrics.index(r["metric"])] = r["mean"]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(fid_metrics)), [m[len("fidelity__") :] for m in fid_metrics])
            ax.set_yticks(range(len(fw_all)), fw_all)
            ax.tick_params(labelsize=6)
            ax.set_title(st, fontsize=7)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax)
        for ax in axes[len(strata_f) :]:
            ax.set_visible(False)
        save_fig(fig, "fidelity.png")

    # (e) capability / over-refusal tax
    tax_rows = [c for m in metric_list if _is_tax(m) for c in contrasts[m]]
    _write_csv(out / "tax.csv", tax_rows)
    if tax_rows:
        files.append("tax.csv")

    # (f) diagnostics: exclusions, truncation/parse rates, heuristic-vs-judge
    excl_rows = _exclusion_rows(summaries)
    bench_rows = _benchmark_rate_rows(summaries, all_rows)
    dis_rows = _heuristic_vs_judge(all_rows)
    for name, diag in (
        ("exclusions", excl_rows),
        ("benchmark_parse_truncation", bench_rows),
        ("heuristic_vs_judge", dis_rows),
    ):
        _write_csv(out / f"{name}.csv", diag)
        if diag:
            files.append(f"{name}.csv")

    # markdown
    md = [f"# ethica report: experiment `{experiment_id}`\n"]
    md.append("## Data\n")
    md.append(
        f"- episodes stored: {counts['episodes']}\n"
        f"- episodes with errors: {counts['errors']} (behavioural metrics scored from events "
        f"recorded before the error; task_success NaN; per-arm table below)\n"
        f"- episodes analysed (with at least one score): {n_analysed}\n"
        f"- distinct tasks: {n_tasks_all}\n"
        f"- baseline: `{baseline}`"
        + (f" (additional: {', '.join(f'`{e}`' for e in extra)})" if extra else "")
        + f"; primary metric: `{primary_metric}`; "
        f"bootstrap {n_boot}, permutations {n_perm}, seed {seed}, permutation unit: {permute_by}\n"
    )
    if not rows:
        md.append("**No scores found for this experiment.**\n")
    if primary_metric not in cells:
        md.append(f"**Primary metric `{primary_metric}` has no scores.**\n")
    md.append("### Scorer used per metric\n")
    md.append(
        _md_table([{"metric": m, "scorer": chosen[m]} for m in metric_list], ["metric", "scorer"])
    )
    warnings = list(dict.fromkeys(warnings))
    if warnings:
        md.append(f"## Warnings: small cells ({len(warnings)})\n")
        md.extend(f"- {w}\n" for w in warnings[:25])
        if len(warnings) > 25:
            md.append(f"- ... and {len(warnings) - 25} more\n")

    md.append("## (a) Per-variant rates\n")
    if primary_metric in rate_rows:
        md.append(
            f"Primary metric `{primary_metric}` (95% cluster-bootstrap CI over tasks). "
            f"Other metrics: `rates__<metric>.csv` / `.png`.\n"
        )
        md.append(
            _md_table(
                rate_rows[primary_metric],
                ["variant", "n", "n_tasks", "mean", "ci_lo", "ci_hi"],
            )
        )
    ref_cols = [
        "stratum",
        "framework",
        "n_pairs",
        "n_tasks",
        "rate",
        "rate_ref",
        "risk_diff",
        "rd_lo",
        "rd_hi",
        "cohens_h",
        "p_perm",
        "p_holm",
    ]
    md.append("## (b) Framework contrasts (within model x safeguards x method)\n")
    md.append(
        "Risk difference = framework minus reference, on matched task x seed pairs; CI is a "
        "task-clustered bootstrap of the paired differences; p is a sign-flip permutation test; "
        "Holm is across frameworks within stratum and reference.\n"
    )
    for m in [primary_metric] if primary_metric in contrasts else []:
        for ref in (baseline, "none", *extra):
            sel = [c for c in contrasts[m] if c["reference"] == ref]
            tag = (
                " (PRIMARY)"
                if ref == baseline
                else " (additional baseline)"
                if ref in extra
                else ""
            )
            md.append(f"### `{m}` vs `{ref}`{tag}\n")
            md.append(_md_table(sel, ref_cols))
    sig = [
        c
        for m in metric_list
        if m != primary_metric
        for c in contrasts[m]
        if c["reference"] == baseline and c["p_holm"] < 0.05
    ]
    md.append(f"### Exploratory metrics: contrasts vs `{baseline}` with Holm p < 0.05\n")
    md.append(_md_table(sig, ["metric", *ref_cols, "p_bh_exploratory"]))
    md.append("All contrasts: `contrasts__<metric>.csv`, forest plots `forest__<metric>.png`.\n")

    md.append("## (c) Safeguards interaction\n")
    md.append(
        "interaction = (framework effect in abliterated) - (framework effect in stock), "
        "computed on task means; positive = framework matters more (raises the metric more) "
        "once safeguards are removed.\n"
    )
    if primary_metric in interaction:
        md.append(
            _md_table(
                interaction[primary_metric],
                [
                    "model_key",
                    "method",
                    "framework",
                    "reference",
                    "n_tasks",
                    "effect_stock",
                    "effect_abliterated",
                    "interaction",
                    "ci_lo",
                    "ci_hi",
                    "p_perm",
                    "p_holm",
                ],
            )
        )
    md.append("## (d) Fidelity (manipulation check)\n")
    if fid_rows:
        md.append(
            "Mean agreement with each framework's expected answers; rows are the variant's "
            "framework. A framework that works should show its highest value on its own "
            "`fidelity__` column. Plot: `fidelity.png`; long-form: `fidelity.csv`.\n"
        )
        md.append(_md_table(fid_rows, ["stratum", "framework", "metric", "n_items", "mean"]))
    else:
        md.append("No `fidelity__*` scores available.\n")
    md.append("## (e) Capability / over-refusal tax\n")
    if tax_rows:
        refs = ", ".join(f"`{r}`" for r in ("none", *extra))
        md.append(f"Difference vs `{baseline}` (and {refs}) for benchmark-style metrics.\n")
        md.append(_md_table(tax_rows, ["metric", "reference", *ref_cols]))
    else:
        md.append("No capability / over-refusal scores available.\n")
    md.append("## (f) Exclusions, truncation and scorer disagreement\n")
    md.append("### Per-arm errors and degraded episodes\n")
    md.append(
        _md_table(
            excl_rows,
            [
                "variant",
                "episodes",
                "errored",
                "context_overflow",
                "retried",
                "truncated",
                "protocol_violation",
                "unclosed_scratchpad",
                "mean_block_tokens",
            ],
        )
    )
    md.append("### Benchmark truncation and unparseable-answer rates per arm\n")
    md.append(
        _md_table(bench_rows, ["variant", "benchmark", "n", "truncated_rate", "unparseable_rate"])
    )
    md.append("### Heuristic vs judge (`over_refusal`, `harmful_compliance`)\n")
    md.append(
        "Rates are over episodes scored by both. The judge is used for the headline numbers "
        "when it ran; large gaps mean the phrase heuristic is biased for that arm.\n"
    )
    md.append(
        _md_table(
            dis_rows,
            ["metric", "variant", "n_both", "heuristic_rate", "judge_rate", "agreement"],
        )
    )
    md.append("## Files\n")
    md.append("\n".join(f"- `{f}`" for f in sorted(set(files))) + "\n")

    report = out / "report.md"
    report.write_text("\n".join(md))
    return report
