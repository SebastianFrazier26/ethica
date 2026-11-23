# Study v1 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one ethical framework into a base (non-instruct) Qwen3-4B checkpoint as its only value-shaping post-training, and evaluate the result with no framework text in context.

**Architecture:** A new `ethica.training` package turns a fixed prompt set plus a local teacher model into per-arm chat datasets; the existing `train_lora` trains one adapter per arm and training seed on the base checkpoint; the existing orchestrator evaluates each adapter as a "trained arm" (method `none`, adapter loaded, no conditioner). The v0 pipeline is untouched.

**Tech Stack:** Python 3.12, uv, mlx-lm 0.32 (LoRA, ChatDataset), pydantic for YAML, pytest, ruff, mypy strict on `src/ethica`.

**Spec:** `docs/study-v1/PROPOSAL.md`; decisions D-61 to D-65 in `docs/DECISIONS.md`.

## Global Constraints

- Starting checkpoint: `Qwen/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`, Apache-2.0. Converted locally to affine 4-bit, group 64 (`q4g64`), as every other local model.
- Teacher: local open-weight model only (`qwen3-4b-abl`). No paid API, no rented GPU (D-64).
- No harmful requests in any arm's training data (D-64). The prompt builder must reject them.
- Pilot arms: `virtue`, `deontology`, `act_utilitarian`, `generic`, `placebo`. 500 dialogues per arm. 3 training seeds per trained arm (default, pending owner, D-65).
- The framework arms do not run until the owner signs off the framework commitments in the proposal (D-64). The `generic` arm needs no sign-off.
- Nothing trained is published: `adapters/`, `models/`, `runs/`, `labels/`, `data/v1/` stay gitignored (proposal section 6).
- Tests never load real models unless marked `slow`. Comments only where load-bearing. One model resident at a time; check for other MLX jobs before any GPU run.
- Existing v0 variant ids, episode ids and adapter hashes must not change (resume depends on them).

## Review Focus

1. A training prompt that slips a harmful request past the filter. Expected: rejected at build time, with a test that the built set shares no item with HarmBench or XSTest and has zero harm-lexicon hits in the dilemma and advice categories beyond an allow-list.
2. A teacher answer that refuses, is cut off at the token limit, or talks about "the reading". Expected: screened out with a recorded reason, never trained on, and the arm's kept count reported.
3. An interrupted generation run. Expected: rerunning continues from the last stored prompt and never duplicates or reorders rows.
4. A trained arm whose adapter directory is missing or half-written at evaluation time. Expected: a clear error before any model loads, not a silent evaluation of the bare base model.
5. Two arms ending up with different prompts or different numbers of training examples. Expected: datasets are built from the intersection of prompts kept in every arm, so arms differ only in the answers.

---

## File structure

| File | Responsibility |
|---|---|
| `src/ethica/types.py` | `Safeguards` gains `"base"` |
| `src/ethica/backends/registry.py` | lineage rule for `base` |
| `scripts/prepare_base_model.py` | convert the base checkpoint, set end-of-turn token, write provenance |
| `configs/models.yaml` | entry `qwen3-4b-base` |
| `src/ethica/training/__init__.py` | package marker |
| `src/ethica/training/prompts.py` | the fixed, seeded training prompt set |
| `src/ethica/training/generate.py` | teacher messages, answer screening, resumable generation |
| `src/ethica/training/dataset.py` | chat-format train/valid files from kept answers |
| `src/ethica/conditioning/lora.py` | `train_lora(..., extra=)` passthrough for `mask_prompt` |
| `src/ethica/config.py`, `src/ethica/orchestrator.py` | `trained_arms` in the experiment config and grid |
| `scripts/v1_pipeline.py` | thin command-line wrapper: `prompts`, `generate`, `dataset`, `train` |
| `configs/experiments/v1-feasibility.yaml` | evaluation config for the feasibility check |
| `tests/test_training_*.py`, `tests/test_trained_arms.py` | tests |

---

### Task 1: Base checkpoint in the registry

**Files:** Modify `src/ethica/types.py`, `src/ethica/backends/registry.py`, `src/ethica/cli.py` (the `models list` filter), `configs/models.yaml`, `.gitignore`. Create `scripts/prepare_base_model.py`. Test `tests/test_registry.py` (append), `tests/test_repo_hygiene.py` (new).

**Interfaces:**
- Produces: `Safeguards = Literal["stock", "abliterated", "base"]`; registry key `qwen3-4b-base` with `local_path: models/qwen3-4b-base-q4g64`.

- [ ] **Step 1: failing tests.** `test_base_model_needs_no_lineage`: a `ModelEntry` with `safeguards="base"` and no `base_of` validates; with `base_of` set it raises `ValueError` matching `"base model must not set base_of"`. `test_no_weights_tracked`: `git ls-files` contains no path ending in `.safetensors`, `.gguf`, `.npz`, and `git check-ignore` succeeds for `adapters/x`, `models/x`, `runs/x`, `labels/x`, `data/v1/x`.
- [ ] **Step 2:** run them; expect failure on the literal and on `data/v1/`.
- [ ] **Step 3:** extend the literal and `_check_lineage`; add `data/v1/` to `.gitignore`; widen the CLI filter to accept `base`.
- [ ] **Step 4:** `scripts/prepare_base_model.py [--chat-format chatml|plain]`: runs `mlx_lm.convert` (q-bits 4, q-group-size 64) from the pinned snapshot into `models/qwen3-4b-base-q4g64/`; for `chatml` sets `eos_token` to `<|im_end|>` in the converted `tokenizer_config.json` (the base checkpoint ships the ChatML template but ends text with `<|endoftext|>`, so a trained model's turn would never stop); for `plain` also replaces `chat_template` with a template built from ordinary tokens (`### System:` / `### User:` / `### Assistant:` blocks, each assistant turn closed by `<|endoftext|>`). Writes `provenance.json` with source repo, sha, quantisation and the tokenizer edits. Reuse `available_gb`/`wait_for_ram` from `scripts/harmonize_quant.py`.
- [ ] **Step 5:** add the `models.yaml` entry (`family: qwen3`, `params_b: 4.0`, `safeguards: base`, `max_context: 32768`, `quant: q4g64`).
- [ ] **Step 6:** `uv run pytest -q`, ruff, mypy; commit.

### Task 2: Training prompt set

**Files:** Create `src/ethica/training/__init__.py`, `src/ethica/training/prompts.py`. Test `tests/test_training_prompts.py`.

**Interfaces:**
- Produces: `Category = Literal["dilemma", "advice", "self", "neutral"]`; `@dataclass(frozen=True) class TrainPrompt: id: str; category: Category; text: str`; `SHARES: dict[Category, float] = {"dilemma": 0.40, "advice": 0.20, "self": 0.15, "neutral": 0.25}`; `build_prompt_set(n: int, seed: int = 0) -> list[TrainPrompt]`; `harm_hits(text: str) -> list[str]`.

Prompts are composed from hand-written parts in the module (situations × question framings for dilemmas and advice; questions × phrasings for self-description; topic lists and generated arithmetic for neutral tasks). Ids are `"<category>-<sha1 of text, 10 hex>"`, so an id never changes when the pool grows.

- [ ] **Step 1: failing tests.**
  - `test_shares_and_size`: `build_prompt_set(500)` has 500 unique texts and per-category counts 200 / 100 / 75 / 125.
  - `test_deterministic_and_seeded`: same seed gives the same list; `build_prompt_set(100)` is a prefix-independent subset of the 500 pool (every id of the smaller set appears in the pool).
  - `test_no_harmful_requests`: for every dilemma and advice prompt `harm_hits(text)` is empty (it uses `ethica.analysis.lexical.is_harm_token` minus an explicit allow-list: `safe`, `safely`, `safety`, `protect*`, `risk*`, `wrong*`, `consent*`, `hurt*`, `suffer*`, `damag*`, `harm*`, `die*`, `death*`, `dying`, `pain*`, `police`, `steal*`, `theft` — words a moral question needs; the list is the single place to review).
  - `test_disjoint_from_eval`: no prompt text equals, or shares a normalised 8-word window with, any HarmBench, XSTest or fidelity-probe item (load through `ethica.benchmarks.registry`).
  - `test_too_large_raises`: asking for more than the pool supports raises `ValueError` naming the short category.
- [ ] **Step 2:** run; expect import failure.
- [ ] **Step 3:** implement. Pool sizes must support n = 2000 for the full study in the dilemma category or the test documents the current ceiling.
- [ ] **Step 4:** tests pass; commit.

### Task 3: Teacher generation

**Files:** Create `src/ethica/training/generate.py`. Test `tests/test_training_generate.py`.

**Interfaces:**
- Consumes: `TrainPrompt`; `RAGConditioner`, `build_index`, `load_corpus`; `ChatBackend`, `GenerationParams`, `Message`.
- Produces:
  - `@dataclass(frozen=True) class ArmSpec: name: str; kind: Literal["ethics", "placebo", "generic"]; corpus: str | None = None; commitments: str | None = None`
  - `teacher_messages(arm: ArmSpec, prompt: TrainPrompt, conditioner: Conditioner | None) -> list[Message]`
  - `screen(text: str, finish_reason: str) -> str | None` (returns a reason or `None` to keep)
  - `@dataclass class GenStats: generated: int; skipped_existing: int; kept: int; rejected: dict[str, int]`
  - `generate_arm(backend: ChatBackend, arm: ArmSpec, prompts: Sequence[TrainPrompt], out_path: Path, params: GenerationParams, conditioner: Conditioner | None = None) -> GenStats`
  - `read_rows(path: Path) -> list[dict[str, Any]]`; row keys: `prompt_id, category, prompt, answer, finish_reason, reject, arm, teacher`.

Teacher instruction (exact copy, system message, after any retrieved block and commitments):
- ethics and placebo arms: `Answer the user in the first person, in your own voice. The outlook in the background reading is your own. Do not mention the reading, do not name any school of thought, and do not say you were asked to take a view. Be direct and specific. Keep the answer under 250 words.`
- generic arm: `Answer the user in the first person, in your own voice. Be direct and specific. Keep the answer under 250 words.`
- `neutral` prompts get the generic instruction and no retrieved block in every arm (format skills must not differ between arms).

Screening reasons: `empty` (under 20 characters), `truncated` (finish reason `length`), `refusal` (matches the harness's refusal heuristic), `meta` (mentions `background reading`, `excerpt`, `the reading`, `as an AI`), `named` (ethics arms only: contains a term from `FRAMEWORK_KEY_TERMS` for that framework).

- [ ] **Step 1: failing tests** (scripted mock backend): generic arm sends no block and the generic instruction; ethics arm with a stub conditioner sends the block before the instruction; neutral prompts get no block in an ethics arm; each screening reason fires on a crafted answer and a clean answer passes; `generate_arm` appends one JSON line per prompt in prompt order; a second call generates nothing and reports `skipped_existing`; a file truncated mid-line is repaired (the partial line dropped) and generation resumes; per-prompt seed is derived from the prompt id, so resumed and uninterrupted runs give identical rows.
- [ ] **Step 2–4:** run, implement, pass.
- [ ] **Step 5:** commit.

### Task 4: Chat dataset

**Files:** Create `src/ethica/training/dataset.py`. Test `tests/test_training_dataset.py`.

**Interfaces:**
- Consumes: `read_rows`.
- Produces: `common_prompt_ids(arm_files: Sequence[Path]) -> list[str]` (prompts kept in every arm, in prompt-set order); `build_chat_dataset(arm_file: Path, out_dir: Path, keep_ids: Sequence[str], seed: int = 0, valid_fraction: float = 0.05) -> dict[str, Any]` writing `train.jsonl`, `valid.jsonl` (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`) and `dataset_meta.json` (`arm, n_train, n_valid, approx_tokens, data_hash, seed`).

- [ ] **Step 1: failing tests:** rejected rows never appear; two arms built with the same `keep_ids` have identical user turns in identical order and the same train/valid split; `common_prompt_ids` drops a prompt rejected in any one arm; valid has at least 4 rows (mlx-lm needs `>= batch_size`); `data_hash` equals `ethica.conditioning.lora.data_hash(out_dir)`; an arm file missing a requested id raises `ValueError`.
- [ ] **Step 2–4:** run, implement, pass. **Step 5:** commit.

### Task 5: Training passthrough and pipeline script

**Files:** Modify `src/ethica/conditioning/lora.py`. Create `scripts/v1_pipeline.py`, `configs/training/v1-pilot.yaml`. Test `tests/test_conditioning.py` (append).

**Interfaces:**
- Produces: `train_lora(model_spec, dataset_dir, out_dir, config=None, extra: Mapping[str, Any] | None = None)`; `extra` keys are merged into the mlx-lm YAML after the fixed keys and recorded in `train_meta.json`. `LoRAConfig` itself is not changed (its fields feed v0 adapter hashes).
- v1 training settings (in `configs/training/v1-pilot.yaml`, read by the script): rank 16, scale 20, 16 layers, batch 2, max_seq_length 1024, learning rate 1e-4, 3 passes over the data (`iters = ceil(3 * n_train / batch)`), `mask_prompt: true` (loss on the assistant turn only), seeds `[0, 1, 2]`. These are starting values; the feasibility check may revise them, and the revision is recorded in `docs/DECISIONS.md`.
- Script commands: `prompts --n --out`, `generate --arm --prompts --out [--limit]`, `dataset --arm-file ... --out-root`, `train --arm --seed`. Paths default under `data/v1/` and `adapters/v1/<arm>-s<seed>/`.

- [ ] **Step 1: failing test** `test_train_lora_extra_passthrough`: with a stub subprocess runner, the written `lora_config.yaml` contains `mask_prompt: true` and `train_meta.json` records it; with `extra=None` the YAML is byte-identical to before (compare against a stored expectation of the v0 keys).
- [ ] **Step 2–4:** run, implement, pass. **Step 5:** commit.

### Task 6: Trained arms in evaluation

**Files:** Modify `src/ethica/config.py`, `src/ethica/orchestrator.py`. Test `tests/test_trained_arms.py`.

**Interfaces:**
- Produces: `class TrainedArm(BaseModel): name: str; kind: Literal["ethics", "placebo", "none"]; adapter: str; model: str`; `ExperimentConfig.trained_arms: list[TrainedArm] = []`.
- `expand_grid` appends, for each trained arm, `Variant(model=<arm.model>, framework=FrameworkSpec(arm.name, arm.kind, arm.name, None), method="none", adapter_path=arm.adapter)`. The variant id hashes the adapter's `train_meta.json` content (or `"pending"` when absent), so retraining an adapter gives a new variant and never reuses old episodes.
- `validate_refs` (run time) raises `ConfigError` naming the arm when `adapters.safetensors` or `train_meta.json` is missing. `ensure_adapter` is never asked to train a trained arm.
- Reporting: the generic arm is declared with `kind: placebo` and named as `report.baseline`; framework arms are `kind: ethics`. One training seed per evaluation config, because the report allows one variant per (model, framework, method).

- [ ] **Step 1: failing tests:** grid contains the arm with method `none` and the adapter path; its conditioner is `NoneConditioner`; a scripted backend factory receives the adapter path for that variant and `None` for the bare base model; changing `train_meta.json` changes the variant id; missing adapter raises `ConfigError` before any backend is created; with no `trained_arms` the grid and every variant id of an existing v0 config (`abl-rag-frameworks.yaml`, corpora permitting, else the mock config) are unchanged; a mock end-to-end run with two trained arms produces a report whose contrast is `<ethics arm>` vs `<generic arm>`.
- [ ] **Step 2–4:** run, implement, pass. **Step 5:** commit.

### Task 7: Feasibility check (GPU; run only when no other MLX job is running)

Question: can LoRA on the base checkpoint, trained on generic dialogues only, produce a model that holds a conversation, stops, and uses the harness's tool protocol?

- [ ] **Step 1:** `uv run python scripts/prepare_base_model.py` (ChatML).
- [ ] **Step 2:** build 500 prompts; generate the `generic` arm with `qwen3-4b-abl` (about 70 min, estimate).
- [ ] **Step 3:** build the dataset; train `generic-s0` (about 20 min, estimate).
- [ ] **Step 4:** evaluate with `configs/experiments/v1-feasibility.yaml`: bare `qwen3-4b-base`, `qwen3-4b` (instruct reference), and trained arm `generic-s0`; all built-in scenarios; `harmbench: 20`, `xstest_unsafe: 20`, `xstest_safe: 15`, `fidelity: 48`; seed 0; judge on.
- [ ] **Step 5:** write `docs/study-v1/FEASIBILITY-<date>.md`. Pass, all three: (a) at least 95% of benchmark answers end by themselves (`benchmark_truncated` ≤ 0.05); (b) answers are on-topic prose on manual reading of 20 samples; (c) zero-tool-call scenarios no more than 1 of 6 above the instruct model's count. If (a) fails with ChatML, rerun steps 1, 3 and 4 with `--chat-format plain` before concluding. If (c) alone fails, the conclusion is "needs shared tool-format examples" (Task 8), not "base model infeasible".
- [ ] **Step 6:** D-56 gate reading (D-63): harmful compliance of `generic-s0` vs bare base, reported, not yet a pass/fail (one seed).

### Later tasks (not part of the overnight run; each needs its own detailing before execution)

- **Task 8: shared neutral block with tool-format examples.** Benign sandbox worlds distinct from the evaluation scenarios, transcripts from the instruct model, identical in every arm.
- **Task 9: critique and revise** against the signed-off commitments.
- **Task 10: internalisation gate:** blind framework identification by the judge, paraphrase consistency, persistence under a contrary system prompt.
- **Task 11: pooling across training seeds** in the analysis (training run as cluster).
- **Task 12: the pilot run** (5 arms × 3 seeds). Blocked on the owner's sign-off of the commitments.

## Self-review notes

- Spec coverage: sections 2 (start, controls) → Tasks 1, 6; section 3 (data) → Tasks 2–4, 9; section 4 (compute) → Task 7 measures it; section 5 (evaluation) → Tasks 6, 10, 11; section 6 (publication) → Task 1 hygiene test; section 8 staging → Tasks 7, 12. The topic-matched placebo arm is covered by `ArmSpec(kind="placebo")`; the second (harm-topic) placebo arm is deferred to the full study.
- Capability set: the harness already has `gsm8k`, `mmlu` and `truthfulqa` benchmarks; the pilot's evaluation config should add them (found while building Task 2). The held-out dilemma set for blind identification is still not covered by any task.
- Prompt pool ceiling: the hand-written pool supports n up to about 530 (limited by 80 self-description prompts). The pilot's 500 fits; the full study's 2,000 needs a larger pool first (add to Task 12's successor).
- Task 6 as built differs from the sketch above in two ways: `TrainedArm.kind` is `ethics | placebo` (a trained `none` arm has no use), and configs with trained arms must list method `none`, so the bare model and the trained arms share a report stratum.
