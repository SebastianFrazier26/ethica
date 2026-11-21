# Changelog

## 2026-11-21 — v0 study

- Study v0 write-up with figures and summary data: `docs/study-v0/` (regenerate with
  `scripts/study_figures.py`).
- Results: RAG framework conditioning on abliterated Qwen3-4B/8B and Granite-4.1-3B — five
  non-consequentialist frameworks reduce harmful compliance vs placebo; utilitarian ones don't.
- LoRA conditioning dropped from v0 (breaks tool-call format or restores refusals; D-56/D-57).
- Experiment configs: refusal check, shakedowns, LoRA guardrail check, multi-framework RAG and
  replication.

## 2026-11-17 — harness v0

- Initial harness: model registry, sandbox, scenarios, benchmarks, prompt/RAG/LoRA conditioning,
  programmatic + judge scoring, SQLite tracking, statistics, reports, CLI, CI.
