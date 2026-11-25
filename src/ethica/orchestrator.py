"""Grid expansion and the generate -> score -> judge -> report phases.

Memory rule (16 GB): exactly one model is resident at any moment. Generation walks models in
config order and loads one backend per (model, adapter); the judge phase loads only the judge.
Every phase is idempotent and resumable from the SQLite store.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ethica.agent_loop import make_episode_id, run_episode, run_single_turn
from ethica.analysis.report import build_report
from ethica.backends.registry import get_entry, make_backend
from ethica.benchmarks import registry as bench_registry
from ethica.benchmarks.base import BenchItem
from ethica.conditioning.corpus import Corpus, corpus_built, corpus_fingerprint, load_corpus
from ethica.conditioning.factory import make_conditioner
from ethica.conditioning.lora import (
    LoRAConfig,
    build_lora_dataset,
    suggested_target_tokens,
    train_lora,
)
from ethica.conditioning.rag import build_and_save_index
from ethica.config import (
    ExperimentConfig,
    resolve_framework,
    resolve_model,
    resolve_scenarios,
    validate_refs,
)
from ethica.scenarios.schema import Scenario
from ethica.scoring.judge import (
    NO_FRAMEWORK,
    Judge,
    applicable_rubrics,
    identify_metric,
    identify_options_for,
    redact_terms_for,
    wants_identification,
)
from ethica.scoring.programmatic import score_programmatic
from ethica.tracking.errors import is_context_overflow
from ethica.tracking.store import Store
from ethica.types import (
    ChatBackend,
    EpisodeResult,
    FrameworkSpec,
    GenerationParams,
    ModelSpec,
    Score,
    Variant,
)

BackendFactory = Callable[[ModelSpec, str | None], ChatBackend]
LoRATrainer = Callable[..., Path]

SECONDS_PER_SCENARIO_EPISODE = 60.0
SECONDS_PER_BENCHMARK_ITEM = 5.0
SECONDS_PER_ADAPTER = 20 * 60.0
LORA_METHODS = ("lora", "lora+prompt")
PRESSURE_SEP = "+under+"


# ---------- offline-first model loading (D-36) ----------


@contextlib.contextmanager
def hf_offline(on: bool) -> Iterator[None]:
    """Toggle HF offline mode. huggingface_hub reads the env var once at import, so the
    constant must be flipped as well."""
    from huggingface_hub import constants

    prev_env = os.environ.get("HF_HUB_OFFLINE")
    prev_const = constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1" if on else "0"
    constants.HF_HUB_OFFLINE = on
    try:
        yield
    finally:
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env
        constants.HF_HUB_OFFLINE = prev_const


def offline_first[T](fn: Callable[[], T], what: str, console: Console) -> T:
    # Offline avoids the HF retry hangs when everything is cached; online is the fallback for
    # a first run that still has to download.
    try:
        with hf_offline(True):
            return fn()
    except Exception as e:
        console.print(
            f"[yellow]{what}: offline load failed ({type(e).__name__}: {str(e)[:120]}); "
            "retrying online (downloads if not cached)[/]"
        )
    with hf_offline(False):
        return fn()


# ---------- grid ----------


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def adapter_hash(
    model: ModelSpec,
    fw: FrameworkSpec,
    corpus_fp: str,
    cfg: LoRAConfig,
    target_tokens: int,
    weights: str = "",
) -> str:
    key = json.dumps(
        [model.hf_repo, weights, fw.name, corpus_fp, asdict(cfg), target_tokens], sort_keys=True
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def weights_key(cfg: ExperimentConfig, model: ModelSpec) -> str:
    """Identity of the weights as the registry resolved them: repo or local path, quantisation,
    and for a local directory the config.json bytes plus weight-file names and sizes."""
    quant = ""
    with contextlib.suppress(KeyError, OSError):
        quant = str(getattr(get_entry(model.key, cfg.models_yaml), "quant", ""))
    parts: list[Any] = [model.hf_repo, quant, model.backend, model.base_of or ""]
    if model.revision:
        parts.append(model.revision)
    local = Path(model.hf_repo)
    if local.is_dir():
        files = sorted(p for p in local.iterdir() if p.suffix in (".safetensors", ".gguf"))
        parts.append([[p.name, p.stat().st_size] for p in files])
        conf = local / "config.json"
        if conf.is_file():
            parts.append(hashlib.sha256(conf.read_bytes()).hexdigest())
        # A changed chat format or tokenizer is a different model to evaluate. Only directories
        # that record such edits are affected, so every v0 variant id stays as it was.
        prov = local / "provenance.json"
        if prov.is_file():
            edits = json.loads(prov.read_text(encoding="utf-8")).get("tokenizer_edits")
            if edits:
                parts.append(_hash(edits))
    return _hash(parts)


def make_variant_id(
    model_key: str,
    framework: str,
    method: str,
    adapter: str | None,
    conditioning: str,
    weights: str = "",
    corpus: str = "",
    gen: str = "",
) -> str:
    """Content-addressed over everything that changes behaviour: resolved weights, corpus
    content, prompt/rag budget and k (in `conditioning`), adapter, generation params. Changing
    any of them yields new variants instead of resuming or averaging with stale episodes."""
    key = json.dumps(
        [model_key, framework, method, adapter or "", conditioning, weights, corpus, gen]
    )
    return "v_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _conditioning_key(cfg: ExperimentConfig, fw: FrameworkSpec, method: str) -> str:
    if fw.kind == "none":
        return ""
    parts: list[str] = []
    if method in ("prompt", "rag", "lora+prompt"):
        parts.append(f"budget={cfg.budget_tokens}")
    if method == "rag":
        parts.append(f"k={cfg.rag_k}")
    return ",".join(parts)


def scenario_content_hash(scenario: Scenario, params: GenerationParams) -> str:
    return _hash([scenario.model_dump(mode="json"), asdict(params)])


def item_content_hash(item: BenchItem, params: GenerationParams) -> str:
    return _hash([asdict(item), asdict(params)])


def scoped_episode_id(experiment_id: str, episode_id: str) -> str:
    """Episode PK is global, so the experiment is folded into the id: two experiments sharing
    a DB can hold the same variant x task x seed without colliding or skipping each other."""
    return "ep_" + _hash([experiment_id, episode_id])


MAX_ATTEMPTS = 3


def _try_corpus(fw: FrameworkSpec) -> Corpus | None:
    if fw.corpus_dir is None or not corpus_built(fw.corpus_dir):
        return None
    return load_corpus(fw.corpus_dir)


def expand_grid(cfg: ExperimentConfig) -> list[Variant]:
    """model x method x framework. Missing corpora are tolerated (dry-run); adapter paths for
    them carry a `pending` marker instead of a content hash.

    The `none` framework is emitted once per method (identical behaviour, distinct variant_id)
    so the report's per-(model, method) strata each contain their no-conditioning reference.
    """
    lora_cfg = cfg.lora_config()
    target = cfg.lora_target_tokens or suggested_target_tokens(lora_cfg)
    fws = {n: resolve_framework(n, cfg.corpora_root) for n in cfg.frameworks}
    corpora = {n: _try_corpus(fw) for n, fw in fws.items()}
    fps = {n: (corpus_fingerprint(c) if c else "pending") for n, c in corpora.items()}
    gen = _hash(cfg.generation.model_dump(mode="json"))
    out: list[Variant] = []
    for key in cfg.models:
        model = resolve_model(key, cfg)
        weights = weights_key(cfg, model)
        for method in cfg.methods:
            for name, fw in fws.items():
                adapter: str | None = None
                ahash: str | None = None
                rag_path: str | None = None
                corpus = corpora[name]
                if method in LORA_METHODS and fw.kind != "none":
                    ahash = (
                        adapter_hash(model, fw, fps[name], lora_cfg, target, weights)
                        if corpus
                        else f"pending-{key}-{name}"
                    )
                    adapter = str(Path(cfg.adapters_dir) / ahash)
                if method == "rag" and fw.kind != "none" and corpus:
                    rag_path = str(cfg.resolved_out_dir / "rag" / f"{name}-{fps[name]}.json")
                out.append(
                    Variant(
                        variant_id=make_variant_id(
                            key,
                            name,
                            method,
                            ahash,
                            _conditioning_key(cfg, fw, method),
                            weights,
                            "" if fw.kind == "none" else fps[name],
                            gen,
                        ),
                        model=model,
                        framework=fw,
                        method=method,
                        adapter_path=adapter,
                        rag_index_path=rag_path,
                    )
                )
    for arm in cfg.trained_arms:
        model = resolve_model(arm.model, cfg)
        meta = Path(arm.adapter) / "train_meta.json"
        # retraining rewrites train_meta.json, so a retrained adapter is a new variant and never
        # resumes or averages with the old one's episodes
        ahash = hashlib.sha256(meta.read_bytes()).hexdigest()[:16] if meta.is_file() else "pending"
        out.append(
            Variant(
                variant_id=make_variant_id(
                    arm.model, arm.name, "none", ahash, "", weights_key(cfg, model), "", gen
                ),
                model=model,
                framework=FrameworkSpec(arm.name, arm.kind, arm.name, None),
                method="none",
                adapter_path=arm.adapter,
            )
        )
        for p in arm.pressure:
            pfw = resolve_framework(p, cfg.corpora_root)
            pcorpus = _try_corpus(pfw)
            name = f"{arm.name}{PRESSURE_SEP}{p}"
            out.append(
                Variant(
                    variant_id=make_variant_id(
                        arm.model,
                        name,
                        "prompt",
                        ahash,
                        _conditioning_key(cfg, pfw, "prompt"),
                        weights_key(cfg, model),
                        corpus_fingerprint(pcorpus) if pcorpus else "pending",
                        gen,
                    ),
                    model=model,
                    # the arm keeps its own label and kind; the corpus is the contrary one
                    framework=replace(pfw, name=name, kind=arm.kind),
                    method="prompt",
                    adapter_path=arm.adapter,
                )
            )
    return out


# ---------- estimate ----------


@dataclass(frozen=True)
class Estimate:
    n_models: int
    n_variants: int
    n_adapters: int
    scenarios: int
    seeds: int
    scenario_episodes: int
    benchmark_items: int
    seconds: float

    def summary(self) -> str:
        h = self.seconds / 3600
        return (
            f"{self.n_models} models, {self.n_variants} variants, {self.n_adapters} LoRA adapters\n"
            f"{self.scenario_episodes} scenario episodes "
            f"({self.scenarios} scenarios x {self.seeds} seeds per variant)\n"
            f"{self.benchmark_items} benchmark items\n"
            f"estimated wall time: {h:.1f} h "
            f"({SECONDS_PER_SCENARIO_EPISODE:.0f} s/episode, {SECONDS_PER_BENCHMARK_ITEM:.0f} "
            f"s/item, {SECONDS_PER_ADAPTER / 60:.0f} min/adapter)"
        )


def estimate(cfg: ExperimentConfig) -> Estimate:
    variants = expand_grid(cfg)
    n_scen = len(resolve_scenarios(cfg))
    n_seeds = len(cfg.seeds)
    adapters = {v.adapter_path for v in variants if v.adapter_path}
    scen_eps = len(variants) * n_scen * n_seeds
    bench_items = len(variants) * sum(cfg.benchmarks.values()) * n_seeds
    secs = (
        scen_eps * SECONDS_PER_SCENARIO_EPISODE
        + bench_items * SECONDS_PER_BENCHMARK_ITEM
        + len(adapters) * SECONDS_PER_ADAPTER
    )
    return Estimate(
        n_models=len(cfg.models),
        n_variants=len(variants),
        n_adapters=len(adapters),
        scenarios=n_scen,
        seeds=n_seeds,
        scenario_episodes=scen_eps,
        benchmark_items=bench_items,
        seconds=secs,
    )


# ---------- adapters ----------


def ensure_adapter(
    cfg: ExperimentConfig,
    variant: Variant,
    console: Console,
    trainer: LoRATrainer = train_lora,
) -> bool:
    """Train the variant's adapter unless train_meta.json exists. Returns True if trained."""
    assert variant.adapter_path is not None
    out = Path(variant.adapter_path)
    if (out / "train_meta.json").is_file():
        return False
    if variant.framework.corpus_dir is None:
        raise ValueError(f"{variant.variant_id}: LoRA variant has no corpus")
    lora_cfg = cfg.lora_config()
    corpus = load_corpus(variant.framework.corpus_dir)
    target = cfg.lora_target_tokens or suggested_target_tokens(lora_cfg)
    ds = out / "data"
    build_lora_dataset(corpus, ds, target, seed=lora_cfg.seed)
    console.print(f"[cyan]training LoRA {variant.model.key} x {variant.framework.name} -> {out}[/]")
    offline_first(
        lambda: trainer(variant.model, ds, out, lora_cfg),
        f"lora {variant.model.key}/{variant.framework.name}",
        console,
    )
    return True


# ---------- generation ----------


@dataclass
class GenerateStats:
    done: int = 0
    skipped: int = 0
    errors: int = 0
    adapters_trained: int = 0
    failed_variants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Task:
    kind: str  # "scenario" | "benchmark"
    episode_id: str
    seed: int
    scenario: Scenario | None = None
    item: BenchItem | None = None
    content_hash: str = ""
    attempts: int = 0  # prior failed attempts; > 0 means an errored row is replaced on save


def load_bench_items(cfg: ExperimentConfig) -> dict[tuple[str, int], list[BenchItem]]:
    """Items depend on the seed (subset sampling, option order), so they are loaded per seed."""
    return {
        (name, seed): bench_registry.get(name).load(n, seed)
        for name, n in cfg.benchmarks.items()
        for seed in cfg.seeds
    }


def _tasks_for(
    cfg: ExperimentConfig,
    variant: Variant,
    scenarios: Sequence[Scenario],
    bench: dict[tuple[str, int], list[BenchItem]],
) -> list[_Task]:
    sp, bp = cfg.generation.scenario_params(), cfg.generation.benchmark_params()
    tasks: list[_Task] = []
    for seed in cfg.seeds:
        for s in scenarios:
            ch = scenario_content_hash(s, sp)
            eid = scoped_episode_id(
                cfg.name,
                make_episode_id(
                    variant.variant_id, s.id, seed, s.followups, cfg.scratchpad, content_hash=ch
                ),
            )
            tasks.append(_Task("scenario", eid, seed, scenario=s, content_hash=ch))
        for name in cfg.benchmarks:
            for item in bench[(name, seed)]:
                ch = item_content_hash(item, bp)
                eid = scoped_episode_id(
                    cfg.name,
                    make_episode_id(
                        variant.variant_id, item.id, seed, scratchpad=False, content_hash=ch
                    ),
                )
                tasks.append(_Task("benchmark", eid, seed, item=item, content_hash=ch))
    return tasks


def _error_episode(variant: Variant, task: _Task, exc: Exception, wall_s: float) -> EpisodeResult:
    task_id = task.scenario.id if task.scenario else (task.item.id if task.item else "?")
    return EpisodeResult(
        episode_id=task.episode_id,
        variant_id=variant.variant_id,
        task_id=task_id,
        task_kind="scenario" if task.kind == "scenario" else "benchmark",
        seed=task.seed,
        transcript=[],
        events=[],
        final_env_state={},
        n_turns=0,
        prompt_tokens=0,
        completion_tokens=0,
        wall_s=wall_s,
        error=f"{type(exc).__name__}: {exc}",
        meta={"stop_reason": "error", "model_id": variant.model.key},
    )


def _progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
    )


def _begin(store: Store, cfg: ExperimentConfig, phase: str) -> None:
    config = cfg.model_dump(mode="json")
    store.create_experiment(cfg.name, config, experiment_id=cfg.name)
    store.record_run(cfg.name, phase, config)


def _needs_run(status: tuple[str | None, int] | None) -> tuple[bool, int]:
    """(run?, prior attempts). Errored episodes are retried on the next run up to MAX_ATTEMPTS,
    except context overflows, which are deterministic and so final."""
    if status is None:
        return True, 0
    error, attempts = status
    if error is None or is_context_overflow(error) or attempts >= MAX_ATTEMPTS:
        return False, attempts
    return True, attempts


def generate(
    cfg: ExperimentConfig,
    backend_factory: BackendFactory = make_backend,
    console: Console | None = None,
    lora_trainer: LoRATrainer = train_lora,
) -> GenerateStats:
    console = console or Console()
    validate_refs(cfg, check_corpora=True)
    stats = GenerateStats()
    scenarios = resolve_scenarios(cfg)
    bench = load_bench_items(cfg)
    variants = expand_grid(cfg)

    store = Store(cfg.resolved_db_url)
    try:
        _begin(store, cfg, "generate")
        for v in variants:
            store.upsert_variant(v)
        prepare_rag_indexes(cfg, variants)

        status = store.episode_status(cfg.name)
        pending: dict[str, list[_Task]] = {}
        for v in variants:
            tasks = _tasks_for(cfg, v, scenarios, bench)
            todo: list[_Task] = []
            for t in tasks:
                run, attempts = _needs_run(status.get(t.episode_id))
                if run:
                    todo.append(replace(t, attempts=attempts))
            stats.skipped += len(tasks) - len(todo)
            if todo:
                pending[v.variant_id] = todo
        total = sum(len(t) for t in pending.values())
        console.print(
            f"{len(variants)} variants; {total} episodes to run, {stats.skipped} already done"
        )
        if not total:
            return stats

        todo_variants = [v for v in variants if v.variant_id in pending]
        seen_adapters: set[str] = set()
        for v in todo_variants:
            if v.adapter_path and v.adapter_path not in seen_adapters:
                seen_adapters.add(v.adapter_path)
                if ensure_adapter(cfg, v, console, lora_trainer):
                    stats.adapters_trained += 1

        with _progress(console) as progress:
            bar = progress.add_task("generate", total=total)
            for model_key in cfg.models:
                groups: dict[str | None, list[Variant]] = {}
                for v in todo_variants:
                    if v.model.key == model_key:
                        groups.setdefault(v.adapter_path, []).append(v)
                for adapter, group in groups.items():
                    _run_group(
                        cfg, store, group, pending, backend_factory, console, progress, bar,
                        stats, adapter,
                    )  # fmt: skip
    finally:
        store.close()
    console.print(
        f"generate done: {stats.done} run, {stats.skipped} resumed-skipped, {stats.errors} errored"
    )
    return stats


def _run_group(
    cfg: ExperimentConfig,
    store: Store,
    group: list[Variant],
    pending: dict[str, list[_Task]],
    backend_factory: BackendFactory,
    console: Console,
    progress: Progress,
    bar: Any,
    stats: GenerateStats,
    adapter: str | None,
) -> None:
    spec = group[0].model
    label = f"{spec.key}" + (f" +adapter {Path(adapter).name}" if adapter else "")
    progress.update(bar, description=f"load {label}")
    try:
        backend = offline_first(lambda: backend_factory(spec, adapter), label, console)
    except Exception as e:
        console.print(f"[red]could not load {label}: {type(e).__name__}: {e}[/]")
        stats.failed_variants.extend(v.variant_id for v in group)
        progress.advance(bar, sum(len(pending[v.variant_id]) for v in group))
        return
    try:
        for v in group:
            _run_variant(
                cfg, store, v, pending[v.variant_id], backend, console, progress, bar, stats
            )
    finally:
        backend.close()


def _run_variant(
    cfg: ExperimentConfig,
    store: Store,
    variant: Variant,
    tasks: list[_Task],
    backend: ChatBackend,
    console: Console,
    progress: Progress,
    bar: Any,
    stats: GenerateStats,
) -> None:
    tag = f"{variant.model.key}|{variant.framework.name}|{variant.method}"
    try:
        conditioner = make_conditioner(variant, cfg.corpora_root, cfg.budget_tokens, cfg.rag_k)
    except Exception as e:
        console.print(f"[red]conditioner for {tag} failed: {type(e).__name__}: {e}[/]")
        stats.failed_variants.append(variant.variant_id)
        progress.advance(bar, len(tasks))
        return
    sp, bp = cfg.generation.scenario_params(), cfg.generation.benchmark_params()
    for t in tasks:
        progress.update(bar, description=f"{tag}")
        t0 = time.perf_counter()
        block_log = getattr(conditioner, "turn_block_tokens", None)
        if block_log is not None:
            block_log.clear()
        try:
            if t.kind == "scenario":
                assert t.scenario is not None
                res = run_episode(
                    backend, conditioner, t.scenario, variant.variant_id, t.seed, sp,
                    scratchpad=cfg.scratchpad, content_hash=t.content_hash,
                )  # fmt: skip
            else:
                assert t.item is not None
                bench = bench_registry.get(t.item.benchmark)
                res = run_single_turn(
                    backend, conditioner, t.item.prompt, bench.system_prompt, bp, t.seed,
                    variant_id=variant.variant_id, task_id=t.item.id,
                    content_hash=t.content_hash,
                )  # fmt: skip
                res.meta["benchmark"] = t.item.benchmark
        except Exception as e:
            res = _error_episode(variant, t, e, time.perf_counter() - t0)
        res.episode_id = t.episode_id
        res.meta["attempts"] = t.attempts + 1
        blocks = block_log or (
            [conditioner.last_block_tokens] if hasattr(conditioner, "last_block_tokens") else []
        )
        if blocks:
            res.meta["block_tokens_mean"] = sum(blocks) / len(blocks)
        if res.error:
            stats.errors += 1
            console.print(f"[red]episode error {tag} {res.task_id} seed={res.seed}: {res.error}[/]")
        try:
            if t.attempts:
                store.delete_episode(t.episode_id)
            if store.save_episode(res, cfg.name):
                stats.done += 1
            else:
                stats.skipped += 1
        except Exception as e:
            stats.errors += 1
            console.print(f"[red]could not save {res.episode_id}: {type(e).__name__}: {e}[/]")
        progress.advance(bar)


# ---------- scoring ----------


@dataclass
class ScoreStats:
    episodes: int = 0
    scores: int = 0
    skipped: int = 0


def _last_assistant(ep: EpisodeResult) -> str:
    return next((m.content for m in reversed(ep.transcript) if m.role == "assistant"), "")


def score(cfg: ExperimentConfig, console: Console | None = None) -> ScoreStats:
    """Programmatic + benchmark scorers. Rescoring replaces the episode's rows from the same
    scorer families, so removed checks or metrics leave nothing stale behind."""
    console = console or Console()
    scenarios = {s.id: s for s in resolve_scenarios(cfg)}
    items: dict[tuple[str, int], dict[str, BenchItem]] | None = None
    stats = ScoreStats()
    store = Store(cfg.resolved_db_url)
    try:
        _begin(store, cfg, "score")
        for ep in store.iter_episodes(cfg.name):
            if ep.task_kind == "scenario":
                scenario = scenarios.get(ep.task_id)
                if scenario is None:
                    stats.skipped += 1
                    continue
                scores = score_programmatic(scenario, ep)
            else:
                if ep.error is not None:
                    stats.skipped += 1
                    continue
                if items is None:
                    items = {k: {i.id: i for i in v} for k, v in load_bench_items(cfg).items()}
                name = str(ep.meta.get("benchmark") or ep.task_id.split(":")[0])
                item = items.get((name, ep.seed), {}).get(ep.task_id)
                if item is None:
                    stats.skipped += 1
                    continue
                scores = [
                    Score(ep.episode_id, m, v, sc, why)
                    for m, v, sc, why in bench_registry.get(name).score(item, _last_assistant(ep))
                ]
                if "truncated" in ep.meta:
                    reasons = ep.meta.get("finish_reasons")
                    scores.append(
                        Score(
                            ep.episode_id,
                            "benchmark_truncated",
                            float(bool(ep.meta["truncated"])),
                            "programmatic",
                            f"finish_reasons={reasons}",
                        )
                    )
            store.save_scores(scores, replace_families=True)
            stats.episodes += 1
            stats.scores += len(scores)
    finally:
        store.close()
    console.print(
        f"score done: {stats.episodes} episodes, {stats.scores} scores, {stats.skipped} skipped"
    )
    return stats


# ---------- judge ----------


@dataclass
class JudgeStats:
    episodes: int = 0
    scores: int = 0
    resumed: int = 0
    errors: int = 0


def judge(
    cfg: ExperimentConfig,
    backend_factory: BackendFactory = make_backend,
    console: Console | None = None,
) -> JudgeStats:
    """Loads ONLY the judge model. Resume is per (episode, rubric): a rubric already scored by
    this judge is skipped, so adding a rubric later judges only that rubric."""
    from ethica.scoring.judge import judge_episodes

    console = console or Console()
    spec = resolve_model(cfg.judge.model, cfg)
    wanted = set(cfg.judge.rubrics) if cfg.judge.rubrics is not None else None
    scenarios = {s.id: s for s in resolve_scenarios(cfg)}
    options = identify_options_for(cfg.corpora_root, cfg.judge.identify_descriptions)
    stats = JudgeStats()
    store = Store(cfg.resolved_db_url)
    try:
        _begin(store, cfg, "judge")
        rows = store.scores_table(cfg.name)
        scorer = f"judge:{spec.key}"
        done: dict[str, set[str]] = {}
        prior: dict[str, dict[str, float]] = {}
        for r in rows:
            if r["scorer"] == scorer:
                done.setdefault(r["episode_id"], set()).add(r["metric"])
            elif not r["scorer"].startswith("judge:") and not math.isnan(r["value"]):
                prior.setdefault(r["episode_id"], {})[r["metric"]] = r["value"]

        todo: list[EpisodeResult] = []
        for ep in store.iter_episodes(cfg.name, include_errors=False):
            _, rubrics = applicable_rubrics(ep, scenarios, prior_scores=prior, only=wanted)
            needed = [r.metric for r in rubrics]
            if options and wants_identification(ep):
                needed.append(identify_metric(NO_FRAMEWORK))
            if not needed:
                continue
            if all(m in done.get(ep.episode_id, ()) for m in needed):
                stats.resumed += 1
            else:
                todo.append(ep)
        if not todo:
            console.print(f"judge done: nothing to do ({stats.resumed} already judged)")
            return stats

        backend = offline_first(lambda: backend_factory(spec, None), spec.key, console)
        try:
            j = Judge(
                backend,
                spec.key,
                redact_terms=redact_terms_for(cfg.corpora_root),
                identify_options=options,
            )
            with _progress(console) as progress:
                bar = progress.add_task(f"judge {spec.key}", total=len(todo))
                for ep in todo:
                    try:
                        out = list(
                            judge_episodes(
                                j, [ep], scenarios, prior_scores=prior, only=wanted, done=done
                            )
                        )
                        store.save_scores(out)
                        stats.scores += len(out)
                        stats.episodes += 1
                    except Exception as e:
                        stats.errors += 1
                        console.print(
                            f"[red]judge failed on {ep.episode_id}: {type(e).__name__}: {e}[/]"
                        )
                    progress.advance(bar)
        finally:
            backend.close()
    finally:
        store.close()
    console.print(f"judge done: {stats.episodes} episodes, {stats.scores} scores")
    return stats


# ---------- report ----------


def report(cfg: ExperimentConfig, console: Console | None = None) -> Path:
    console = console or Console()
    store = Store(cfg.resolved_db_url)
    try:
        _begin(store, cfg, "report")
        r = cfg.report
        path = build_report(
            store,
            cfg.name,
            cfg.resolved_out_dir / "report",
            variant_ids={v.variant_id for v in expand_grid(cfg)},
            baseline=r.baseline,
            extra_baselines=r.extra_baselines,
            min_tasks=r.min_tasks,
            n_boot=r.n_boot,
            n_perm=r.n_perm,
            permute_by=r.permute_by,
        )
    finally:
        store.close()
    console.print(f"report: {path}")
    return path


def prepare_rag_indexes(cfg: ExperimentConfig, variants: Sequence[Variant]) -> None:
    """Build+save each distinct RAG index once; make_conditioner loads it if present."""
    for v in variants:
        p = v.rag_index_path
        if p and not Path(p).is_file() and v.framework.corpus_dir:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            build_and_save_index(load_corpus(v.framework.corpus_dir), p)


def load_score_objects(store: Store, experiment_id: str) -> list[Score]:
    """Scores with rationale (scores_table omits it); the label exporter needs the judge's."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ethica.tracking.models import Episode, ScoreRow

    with Session(store.engine) as s:
        rows = s.execute(
            select(ScoreRow)
            .join(Episode, Episode.episode_id == ScoreRow.episode_id)
            .where(Episode.experiment_id == experiment_id)
        ).scalars()
        return [
            Score(
                r.episode_id,
                r.metric,
                math.nan if r.value is None else r.value,
                r.scorer,
                r.rationale or "",
            )
            for r in rows
        ]
