"""`ethica` command line. Thin: argument validation and printing; logic lives in the modules."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ethica.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    resolve_framework,
    resolve_model,
    validate_refs,
)

app = typer.Typer(help="ethica: ethical-framework conditioning vs misalignment eval harness.")
models_app = typer.Typer(help="Model registry.")
corpus_app = typer.Typer(help="Conditioning corpora.")
lora_app = typer.Typer(help="LoRA adapters.")
label_app = typer.Typer(help="Human labels for judge validation.")
scen_app = typer.Typer(help="Agentic scenarios.")
bench_app = typer.Typer(help="Static benchmarks.")
for name, sub in [
    ("models", models_app),
    ("corpus", corpus_app),
    ("lora", lora_app),
    ("label", label_app),
    ("scenarios", scen_app),
    ("benchmarks", bench_app),
]:
    app.add_typer(sub, name=name)

console = Console()
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.+\-]*$")


class Phase(StrEnum):
    all = "all"
    generate = "generate"
    score = "score"
    judge = "judge"
    report = "report"


@contextmanager
def _friendly() -> Iterator[None]:
    """Turn expected user errors into one red line and exit code 2 instead of a traceback."""
    try:
        yield
    except (ConfigError, KeyError, ValueError, FileNotFoundError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        console.print(f"[red]error:[/] {msg}")
        raise typer.Exit(2) from e


def _slug(value: str) -> str:
    if not _SLUG.fullmatch(value):
        raise typer.BadParameter(f"{value!r} is not a lowercase slug")
    return value


ConfigArg = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Experiment YAML.")
]


def _load(config: Path, *, check_corpora: bool = False) -> ExperimentConfig:
    cfg = load_config(config)
    validate_refs(cfg, check_corpora=check_corpora)
    return cfg


# ---------- models ----------


@models_app.command("list")
def models_list(
    family: Annotated[str | None, typer.Option()] = None,
    safeguards: Annotated[str | None, typer.Option(help="stock | abliterated | base")] = None,
) -> None:
    from ethica.backends.registry import load_entries

    if safeguards not in (None, "stock", "abliterated", "base"):
        raise typer.BadParameter("safeguards must be stock, abliterated or base")
    with _friendly():
        t = Table("key", "family", "params_b", "safeguards", "quant", "size_gb", "lora", "backend")
        for e in load_entries().values():
            if (family and e.family != family) or (safeguards and e.safeguards != safeguards):
                continue
            t.add_row(
                e.key, e.family, f"{e.params_b:g}", e.safeguards, e.quant, f"{e.size_gb:g}",
                "y" if e.lora_ok else "n", e.backend,
            )  # fmt: skip
        console.print(t)


@models_app.command("pull")
def models_pull(keys: Annotated[list[str], typer.Argument(help="Model keys.")]) -> None:
    """Download model weights into the HF cache."""
    from ethica.backends.registry import pull

    with _friendly():
        for k in keys:
            console.print(f"pulling {k} ...")
            console.print(pull(k))


# ---------- corpus ----------


@corpus_app.command("list")
def corpus_list(root: Annotated[Path, typer.Option()] = Path("corpora")) -> None:
    from ethica.conditioning.corpus import list_frameworks, load_corpus

    with _friendly():
        t = Table("name", "kind", "docs", "~tokens", "display_name")
        for fw in list_frameworks(root):
            if fw.corpus_dir is None:
                t.add_row(fw.name, fw.kind, "-", "-", fw.display_name)
                continue
            c = load_corpus(fw.corpus_dir)
            t.add_row(
                fw.name, fw.kind, str(len(c.documents)), str(c.total_tokens()), fw.display_name
            )
        console.print(t)


@corpus_app.command("build-placebo")
def corpus_build_placebo(
    out: Annotated[Path, typer.Option(help="Output dir; its name becomes the corpus name.")] = Path(
        "corpora/placebo"
    ),
    tokens: Annotated[int, typer.Option(min=1000, help="Approx token target.")] = 200_000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Stream Simple English Wikipedia and keep topic-filtered, non-normative articles."""
    from ethica.conditioning.placebo import build_placebo_corpus

    with _friendly():
        _slug(out.name)
        m = build_placebo_corpus(out, tokens, seed)
        console.print(f"wrote {len(m.sources)} articles to {out}")


@corpus_app.command("check")
def corpus_check(
    framework: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("corpora"),
) -> None:
    """Validate one framework's manifest and files; print size."""
    from ethica.conditioning.corpus import load_corpus

    with _friendly():
        _slug(framework)
        fw = resolve_framework(framework, root)
        if fw.corpus_dir is None:
            console.print("none: nothing to check")
            return
        c = load_corpus(fw.corpus_dir)
        console.print(
            f"[green]ok[/] {fw.name} ({fw.kind}): {len(c.documents)} docs, "
            f"~{c.total_tokens()} tokens\nsummary: {fw.summary}"
        )


# ---------- lora ----------


@lora_app.command("train")
def lora_train(
    model: Annotated[str, typer.Option()],
    framework: Annotated[str, typer.Option()],
    corpora_root: Annotated[Path, typer.Option()] = Path("corpora"),
    adapters_dir: Annotated[Path, typer.Option()] = Path("adapters"),
    models_yaml: Annotated[Path | None, typer.Option(exists=True)] = None,
) -> None:
    """Train (or reuse) the cached adapter for one model x framework."""
    from ethica.orchestrator import ensure_adapter, expand_grid

    with _friendly():
        _slug(framework)
        cfg = ExperimentConfig(
            name="lora-train",
            models=[model],
            frameworks=[framework],
            methods=["lora"],
            corpora_root=str(corpora_root),
            adapters_dir=str(adapters_dir),
            models_yaml=str(models_yaml) if models_yaml else None,
        )
        validate_refs(cfg, check_corpora=True)
        resolve_model(model, cfg)
        variant = next(v for v in expand_grid(cfg) if v.adapter_path)
        trained = ensure_adapter(cfg, variant, console)
        console.print(f"{'trained' if trained else 'reused'}: {variant.adapter_path}")


# ---------- run / phases ----------


@app.command()
def run(
    config: ConfigArg,
    phase: Annotated[Phase, typer.Option(help="Run one phase instead of all.")] = Phase.all,
    dry_run: Annotated[bool, typer.Option(help="Print grid size and time estimate only.")] = False,
) -> None:
    """generate -> score -> [judge] -> report."""
    from ethica import orchestrator as orch

    with _friendly():
        cfg = _load(config, check_corpora=not dry_run and phase in (Phase.all, Phase.generate))
        if dry_run:
            console.print(f"[bold]{cfg.name}[/] (dry run)\n{orch.estimate(cfg).summary()}")
            return
        if phase in (Phase.all, Phase.generate):
            orch.generate(cfg, console=console)
        if phase in (Phase.all, Phase.score):
            orch.score(cfg, console=console)
        if phase == Phase.judge or (phase == Phase.all and cfg.judge.enabled):
            orch.judge(cfg, console=console)
        if phase in (Phase.all, Phase.report):
            orch.report(cfg, console=console)


@app.command()
def score(config: ConfigArg) -> None:
    """Programmatic + benchmark scoring of stored episodes (idempotent)."""
    from ethica import orchestrator as orch

    with _friendly():
        orch.score(_load(config), console=console)


@app.command()
def judge(config: ConfigArg) -> None:
    """Judge pass (loads only the judge model); runs even if judge.enabled is false."""
    from ethica import orchestrator as orch

    with _friendly():
        orch.judge(_load(config), console=console)


@app.command()
def report(config: ConfigArg) -> None:
    from ethica import orchestrator as orch

    with _friendly():
        orch.report(_load(config), console=console)


# ---------- label ----------


@label_app.command("export")
def label_export(
    config: ConfigArg,
    out: Annotated[Path, typer.Option(help=".jsonl or .csv; key file written beside it.")],
    metric: Annotated[str, typer.Option()],
    n: Annotated[int, typer.Option(min=1)] = 50,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Blind labelling sheet over judged episodes (judge verdicts go to a separate key file)."""
    from ethica.backends.mock import MockBackend
    from ethica.config import resolve_scenarios
    from ethica.orchestrator import load_score_objects
    from ethica.scoring.agreement import export_label_sample
    from ethica.scoring.judge import Judge
    from ethica.tracking.store import Store

    if out.suffix not in (".jsonl", ".csv"):
        raise typer.BadParameter("out must end in .jsonl or .csv")
    with _friendly():
        cfg = _load(config)
        store = Store(cfg.resolved_db_url)
        try:
            episodes = list(store.iter_episodes(cfg.name, include_errors=False))
            scores = load_score_objects(store, cfg.name)
        finally:
            store.close()
        scen = {s.id: s for s in resolve_scenarios(cfg)}
        j = Judge(MockBackend(["-"]), cfg.judge.model)  # render only; never generates
        rows = export_label_sample(
            j, episodes, scores, n, seed, out, metric=metric, scenario_lookup=scen
        )
        console.print(f"wrote {rows} items to {out} (+ key file)")


@label_app.command("import")
def label_import(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate a filled-in label sheet and summarise it."""
    from collections import Counter

    from ethica.scoring.agreement import import_labels

    with _friendly():
        labels = import_labels(path)
        console.print(f"{len(labels)} labelled items: {dict(Counter(labels.values()))}")


@label_app.command("agreement")
def label_agreement(
    labels: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    key: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Human labels vs judge verdicts: percent agreement and Cohen's kappa."""
    from ethica.scoring.agreement import agreement_report, import_key, import_labels

    with _friendly():
        rep = agreement_report(import_labels(labels), import_key(key))
        t = Table("statistic", "value")
        for k, v in rep.items():
            t.add_row(k, f"{v:.3f}")
        console.print(t)


# ---------- scenarios / benchmarks ----------


@scen_app.command("list")
def scenarios_list(category: Annotated[str | None, typer.Option()] = None) -> None:
    from ethica.scenarios.registry import list_scenarios

    t = Table("id", "category", "bait", "max_turns", "title")
    for s in list_scenarios(category=category):
        t.add_row(s.id, s.category, "y" if s.framework_bait else "", str(s.max_turns), s.title)
    console.print(t)


@scen_app.command("show")
def scenarios_show(scenario_id: Annotated[str, typer.Argument()]) -> None:
    from ethica.scenarios.registry import get

    with _friendly():
        console.print(yaml.safe_dump(get(scenario_id).model_dump(mode="json"), sort_keys=False))


@bench_app.command("list")
def benchmarks_list() -> None:
    from ethica.benchmarks import registry

    t = Table("name", "default_n")
    for name in registry.names():
        t.add_row(name, str(registry.get(name).default_n))
    console.print(t)


@bench_app.command("prefetch")
def benchmarks_prefetch() -> None:
    """Download every pinned benchmark file into data/cache/."""
    from ethica.benchmarks.prefetch import prefetch_all

    for p in prefetch_all():
        console.print(str(p))


if __name__ == "__main__":
    app()
