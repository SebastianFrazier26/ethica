# ethica

A fully-local research harness for one question: **does training or conditioning a
small language model on a particular ethical framework make it more or less
misaligned?**

Small open-weight models (stock and with safety training removed) are conditioned on
literature from one normative framework — deontology, act/rule utilitarianism, virtue
ethics, theological ethics, and others — or on a placebo corpus, then dropped into a
simulated sandbox with agentic tasks that contain temptations to subvert oversight,
deceive, resist shutdown, or hack their reward. Every action is logged; behaviour is
scored programmatically and by a local judge model; results are compared with proper
statistics.

Runs entirely on an Apple Silicon Mac (MLX). No cloud APIs.

- **Study v0 (design, results, figures): [`docs/study-v0/README.md`](docs/study-v0/README.md)**
- Design: [`docs/DESIGN.md`](docs/DESIGN.md)
- Decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md)

```bash
uv sync
uv run pytest
uv run ethica --help
```

## Quickstart

```bash
uv sync
uv run ethica benchmarks prefetch            # pinned datasets -> data/cache/
uv run ethica models pull tiny               # 0.5B smoke model (~0.3 GB) into the HF cache
uv run ethica run configs/experiments/smoke.yaml        # generate -> score -> report, ~90 s
uv run ethica run configs/experiments/smoke-judge.yaml --phase judge   # exercise the judge path
uv run ethica run configs/experiments/pilot.yaml --dry-run             # grid size + time estimate
```

`configs/experiments/smoke-mock.yaml` runs the same pipeline on a scripted mock backend
(no weights). Smoke results are plumbing checks, not science.

Where outputs land (`<out>` = `runs/<name>/` unless the config sets `out_dir`):

- `<out>/ethica.db` - SQLite: every episode, message, env event and score (`db_url` overrides)
- `<out>/report/report.md` - rates, framework-vs-placebo contrasts, plots and CSVs beside it
- `adapters/<hash>/` - cached LoRA adapters (reused when `train_meta.json` exists)
- `<out>/rag/` - cached BM25 indexes

Phases are resumable: rerunning a config skips episodes already in the DB, and `score` /
`judge` are idempotent. Other commands: `ethica --help` (`models`, `corpus`, `lora train`,
`label export|import|agreement`, `scenarios list|show`, `benchmarks list|prefetch`).

Status: pre-alpha. First exploratory study complete (see `docs/study-v0/`). Study v1, which trains each framework into a base model's weights instead of supplying it in context, is being built (`docs/study-v1/PROPOSAL.md`); scenarios and curated corpora in progress.

License: MIT.
