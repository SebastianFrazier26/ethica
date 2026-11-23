# Changelog

## 2026-11-23

- v1 internalisation gate (`docs/study-v1/GATE.md`): `open_dilemmas` benchmark, blind framework
  identification by the judge (`identified__<framework>`), phrasing and order consistency,
  persistence variants (`pressure:` on a trained arm), `scripts/v1_gate.py`. Control runs: v0's
  RAG arms are identified as their own framework 90–98% of the time; unconditioned models read
  as act-utilitarian about half the time.
- Arm contrasts pooled over training seeds: `ethica.analysis.pooling`, `scripts/v1_pool.py`.
- v1 feasibility check (`docs/study-v1/FEASIBILITY-2026-11-23.md`): a LoRA-trained base checkpoint
  converses and stops with a plain-token chat format (ChatML fails), tool use still fails, and the
  bare base checkpoint already refuses most harmful requests (D-66, D-67).
- Second run the same night: tool-protocol training examples shared by every arm
  (`ethica.training.toolformat`), tool tags spelled as ordinary text for the base model, reworded
  teacher instruction. Tool use now works (0 of 6 scenarios without a tool call). Baseline table
  with `gsm8k`/`mmlu` for the instruct model, bare base and generic arm in the feasibility report;
  config `configs/experiments/v1-baseline.yaml`.

## 2026-11-22

- Study v1 chosen by the owner (D-61 to D-65): train each framework into a base checkpoint as its
  only post-training and evaluate with no framework in context. Added: `ethica.training`
  (prompt set, teacher generation, chat datasets, base-model chat format), `trained_arms` in
  experiment configs, base checkpoint `qwen3-4b-base` in the registry, `scripts/v1_pipeline.py`,
  `scripts/prepare_base_model.py`, draft framework commitments (awaiting owner sign-off), and an
  implementation plan (`docs/superpowers/plans/2026-11-22-study-v1-pilot.md`).
- Draft proposal for study v1 (framework trained into the weights instead of supplied in context):
  `docs/study-v1/PROPOSAL.md`. Draft for owner review; nothing decided, trained or downloaded.
- Lexical-confound control for the v0 RAG result (D-58). Offline analysis of every retrieved
  block: `scripts/lexical_analysis.py` → `docs/study-v0/data/lexical_*.csv`, measures in
  `ethica.analysis.lexical` (harm lexicon, request overlap, Spearman, logistic regression).
  Harm-vocabulary density does not rank the framework effects.
- New placebo corpus `placebo_harmvocab` (manifest only; Simple English Wikipedia on poisons,
  weapons, drugs, disease, crime, accidents and chemical hazards, no moral vocabulary, ~204k tokens).
- Configs `abl-rag-lexical.yaml` / `abl-rag-lexical-replicate.yaml` add it to the v0 grids and
  reuse the stored episodes (`scripts/seed_lexical_runs.py`); `scripts/lexical_contrasts.py`
  computes contrasts against it.
- Control result, all three models: `placebo_harmvocab` does not lower harmful compliance (0.58 /
  0.68 / 0.65 vs geology placebo 0.46 / 0.60 / 0.62). All 15 non-consequentialist contrasts stay
  negative against it (14/15 Holm-significant); the utilitarian arms become −0.10 to −0.14 on both
  Qwen models. `docs/study-v0/data/contrasts_vs_placebo_harmvocab.csv`.
- Report: `report.extra_baselines` adds contrasts against further reference arms (D-59,
  `docs/ANALYSIS.md`); enabled for `placebo_harmvocab` in the lexical configs.
- Figures: `scripts/study_figures.py` adds the harm-topic placebo to the rates figure and writes
  `figures/forest_harmful_compliance_vs_harmvocab.png`. Primary forest and baseline unchanged.
- Study write-up now reports both placebos side by side (owner decision D-60): TL;DR, design and
  results rewritten, utilitarian frameworks described as much weaker and null against the geology
  placebo; forest titles updated. `placebo_harmvocab` corpus accepted as is, with its 40 flagged
  titles noted as a caveat. D-57 owner-confirmed.
- Corpus-review aid: `scripts/retrieval_review.py` reports which sources fill the retrieved excerpts
  on the harmful prompts (tables in `docs/CORPORA.md`).

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
