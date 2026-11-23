"""Figures + summary CSVs for the v0 study write-up (docs/study-v0/).

Reads the local run DBs / report CSVs (runs/ is gitignored), writes PNGs and CSVs that are tracked.
Run: uv run python scripts/study_figures.py
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "study-v0"
FIG = OUT / "figures"
DATA = OUT / "data"

# validated categorical slots 1-3 (dataviz reference palette, light mode); aqua < 3:1 contrast,
# so every chart also carries marker shape + legend, and the README carries the numbers.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "D"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

ETHICS = [
    "virtue",
    "deontology",
    "care",
    "contractualism",
    "divine_command",
    "rule_utilitarian",
    "act_utilitarian",
]
LABEL = {
    "virtue": "Virtue",
    "deontology": "Deontology",
    "care": "Care",
    "contractualism": "Contractualism",
    "divine_command": "Divine command",
    "rule_utilitarian": "Rule utilitarian",
    "act_utilitarian": "Act utilitarian",
    "placebo": "Placebo (geology)",
    "placebo_harmvocab": "Placebo (harm topics)",
    "none": "None",
}
HARMVOCAB = "placebo_harmvocab"
# D-58: the control runs hold the v0 episodes plus the harm-vocabulary placebo arm
CONTROL_RUN = {
    "abl-rag-frameworks": "abl-rag-lexical",
    "abl-rag-replicate": "abl-rag-lexical-replicate",
}
MODELS = [
    ("qwen3-4b-abl", "Qwen3-4B (abl.)", "abl-rag-frameworks"),
    ("qwen3-8b-abl", "Qwen3-8B (abl.)", "abl-rag-replicate"),
    ("granite-4.1-3b-abl", "Granite-4.1-3B (abl.)", "abl-rag-replicate"),
]
PAIRS = [
    ("qwen3-1.7b", "Qwen3-1.7B"),
    ("qwen3-4b", "Qwen3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
    ("llama-3.2-3b", "Llama-3.2-3B"),
    ("gemma-3-4b", "Gemma-3-4B"),
    ("smollm3-3b", "SmolLM3-3B"),
    ("phi-4-mini", "Phi-4-mini"),
    ("granite-4.1-3b", "Granite-4.1-3B"),
]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _db(run: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{RUNS / run / 'ethica.db'}?mode=ro", uri=True)


def _write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    with (DATA / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def control_runs_built() -> bool:
    return all(
        (RUNS / run / "report" / "contrasts__harmful_compliance.csv").is_file()
        for run in CONTROL_RUN.values()
    )


def contrasts(reference: str = "placebo") -> list[dict[str, str]]:
    runs = CONTROL_RUN.values() if reference == HARMVOCAB else CONTROL_RUN.keys()
    rows: list[dict[str, str]] = []
    for run in runs:
        with (RUNS / run / "report" / "contrasts__harmful_compliance.csv").open() as f:
            rows += [r for r in csv.DictReader(f) if r["reference"] == reference]
    return rows


def significant(p_holm: float, lo: float, hi: float) -> bool:
    """Holm p < 0.05. A placebo-vs-placebo row sits outside the Holm family (NaN), so it
    falls back to whether its 95% CI excludes 0."""
    if p_holm != p_holm:
        return lo > 0 or hi < 0
    return p_holm < 0.05


def fig_forest(
    rows: list[dict[str, str]],
    arms: list[str],
    stem: str,
    title: str,
    ref_label: str,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2 + 0.5 * (len(arms) - len(ETHICS))), facecolor=SURFACE)
    _style(ax)
    n = len(MODELS)
    out = []
    for i, (key, label, _) in enumerate(MODELS):
        ys, xs, lo, hi, sig = [], [], [], [], []
        for j, fw in enumerate(arms):
            r = next(r for r in rows if r["model_key"] == key and r["framework"] == fw)
            y = j + (i - (n - 1) / 2) * 0.24
            rd, a, b, p = (float(r[k]) for k in ("risk_diff", "rd_lo", "rd_hi", "p_holm"))
            ys.append(y), xs.append(rd), lo.append(rd - a), hi.append(b - rd)
            sig.append(significant(p, a, b))
            p_txt = "" if p != p else f"{p:.4f}"
            out.append([key, fw, f"{rd:.3f}", f"{a:.3f}", f"{b:.3f}", p_txt, r["n_pairs"]])
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="none", ecolor=SERIES[i], elinewidth=2, capsize=0)
        for x, y, s in zip(xs, ys, sig, strict=True):
            ax.plot(
                x,
                y,
                MARKERS[i],
                ms=8,
                color=SERIES[i] if s else SURFACE,
                markeredgecolor=SERIES[i],
                markeredgewidth=2,
            )
        ax.plot([], [], MARKERS[i], ms=8, color=SERIES[i], label=label)
    ax.axvline(0, color=INK2, linewidth=1)
    for boundary in (4.5, 6.5)[: 1 + (len(arms) > len(ETHICS))]:
        ax.axhline(boundary, color=GRID, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(arms)), [LABEL[f] for f in arms], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Change in harmful compliance vs {ref_label} (risk difference, 95% CI)", color=INK2
    )
    ax.set_title(
        title,
        color=INK,
        fontsize=11,
        loc="left",
        pad=26,
    )
    ax.text(
        0.0,
        -0.17,
        "Filled marker = Holm-corrected p < 0.05; hollow = not significant.\n"
        f"Left of 0 = safer than {ref_label}." + note,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=INK2,
    )
    ax.legend(
        frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK
    )
    fig.tight_layout()
    fig.savefig(FIG / f"forest_harmful_compliance{stem}.png", dpi=160)
    plt.close(fig)
    _write_csv(
        f"contrasts_vs_{rows[0]['reference']}.csv",
        ["model", "framework", "risk_diff", "ci_lo", "ci_hi", "p_holm", "n_pairs"],
        out,
    )


def _rates(run: str, key: str) -> dict[str, float]:
    q = """select v.framework, avg(s.value) from scores s join episodes e using(episode_id)
           join variants v on v.variant_id=e.variant_id
           where s.metric='harmful_compliance' and s.scorer like 'judge%' and v.model_key=?
           group by 1"""
    with _db(run) as c:
        return {f: float(x) for f, x in c.execute(q, (key,))}


def fig_rates(with_control: bool) -> None:
    order = ["none", "placebo", *([HARMVOCAB] if with_control else []), *ETHICS]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    _style(ax)
    out = []
    for i, (key, label, run) in enumerate(MODELS):
        r = _rates(CONTROL_RUN[run] if with_control else run, key)
        ys = [j + (i - 1) * 0.24 for j in range(len(order))]
        xs = [r[f] for f in order]
        ax.plot(xs, ys, MARKERS[i], ms=8, color=SERIES[i], label=label, linestyle="none")
        out += [[key, f, f"{r[f]:.3f}"] for f in order]
    ax.axhline(len(order) - len(ETHICS) - 0.5, color=GRID, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(order)), [LABEL[f] for f in order], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel(
        "Harmful compliance rate (judge; HarmBench + XSTest-unsafe; lower = safer)", color=INK2
    )
    ax.set_title(
        "Harmful compliance by conditioning, abliterated models (RAG)",
        color=INK,
        fontsize=11,
        loc="left",
        pad=26,
    )
    ax.legend(
        frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK
    )
    fig.tight_layout()
    fig.savefig(FIG / "rates_harmful_compliance.png", dpi=160)
    plt.close(fig)
    _write_csv("rates_harmful_compliance.csv", ["model", "framework", "rate"], out)


def fig_refusal_check() -> None:
    q = """select v.model_key, avg(s.value) from scores s join episodes e using(episode_id)
           join variants v on v.variant_id=e.variant_id
           where s.metric='harmful_compliance' and s.scorer like 'judge%'
             and e.task_id like 'harmbench%' group by 1"""
    with _db("refusal-check") as c:
        r = {k: float(x) for k, x in c.execute(q)}
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    out = []
    for j, (key, _label) in enumerate(PAIRS):
        s, a = r[key], r[f"{key}-abl"]
        ax.plot([s, a], [j, j], color=GRID, linewidth=2, zorder=1)
        ax.plot(s, j, MARKERS[0], ms=8, color=SERIES[0], zorder=2)
        ax.plot(a, j, MARKERS[1], ms=8, color=SERIES[1], zorder=2)
        out.append([key, f"{s:.3f}", f"{a:.3f}"])
    ax.plot([], [], MARKERS[0], ms=8, color=SERIES[0], label="Stock")
    ax.plot([], [], MARKERS[1], ms=8, color=SERIES[1], label="Abliterated")
    ax.set_yticks(range(len(PAIRS)), [p[1] for p in PAIRS], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("HarmBench harmful compliance (judge, n=40 per model)", color=INK2)
    ax.set_title(
        "Abliteration check: stripped models comply far more",
        color=INK,
        fontsize=11,
        loc="left",
        pad=26,
    )
    ax.legend(
        frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK
    )
    fig.tight_layout()
    fig.savefig(FIG / "refusal_check.png", dpi=160)
    plt.close(fig)
    _write_csv("refusal_check_harmbench.csv", ["family", "stock", "abliterated"], out)


def fig_fidelity() -> None:
    rows = ["none", "placebo", *ETHICS]
    q = """select v.framework, s.metric, avg(s.value) from scores s
           join episodes e using(episode_id)
           join variants v on v.variant_id=e.variant_id
           where s.metric like 'fidelity__%' and v.model_key='qwen3-4b-abl' group by 1, 2"""
    with _db("abl-rag-frameworks") as c:
        m = {(f, met.removeprefix("fidelity__")): float(x) for f, met, x in c.execute(q)}
    grid = [[m.get((r, col), float("nan")) for col in ETHICS] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ETHICS)), [LABEL[c] for c in ETHICS], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [LABEL[r] for r in rows])
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, r in enumerate(grid):
        for j, v in enumerate(r):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if v > 0.6 else INK,
            )
    for i, r in enumerate(rows):
        if r in ETHICS:
            j = ETHICS.index(r)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2))
    ax.set_xlabel("Fidelity probe: agreement with each framework's answers", color=INK2)
    ax.set_ylabel("Conditioned on", color=INK2)
    ax.set_title(
        "Manipulation check (Qwen3-4B abl., RAG): outlined = own framework",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "fidelity_heatmap.png", dpi=160)
    plt.close(fig)
    _write_csv(
        "fidelity_qwen3-4b-abl.csv",
        ["conditioned_on", *ETHICS],
        [[r, *(f"{v:.3f}" for v in g)] for r, g in zip(rows, grid, strict=True)],
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig_forest(
        contrasts(),
        ETHICS,
        "",
        "Against the geology placebo: five frameworks lower, utilitarian null",
        "the geology placebo",
    )
    with_control = control_runs_built()
    if with_control:
        fig_forest(
            contrasts(HARMVOCAB),
            [*ETHICS, "placebo"],
            "_vs_harmvocab",
            "Against the harm-topic placebo: all lower on Qwen, utilitarian weakest",
            "the harm-topic placebo",
            " Geology row: no Holm correction, filled = 95% CI excludes 0.",
        )
    fig_rates(with_control)
    fig_refusal_check()
    fig_fidelity()
    print(f"wrote {FIG} and {DATA}")


if __name__ == "__main__":
    main()
