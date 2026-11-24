# ethica — agent notes

Research harness: does ethical-framework conditioning change misalignment in small local LLMs.
Read `docs/DESIGN.md` (architecture), `docs/DECISIONS.md` (why), `docs/METRICS.md` (score names).

## Commands
- `uv sync` — install
- `uv run pytest` — fast suite (excludes `slow`); `uv run pytest -m slow` loads real models / downloads
- `uv run ruff check . && uv run ruff format --check .` — lint
- `uv run mypy` — types (strict on `src/ethica`)
- `uv run ethica --help` — CLI
- `uv run ethica run configs/experiments/smoke-mock.yaml` — full pipeline on the mock backend (instant)
- `uv run ethica run configs/experiments/smoke.yaml` — same with the real 0.5B `tiny` MLX model (~90 s)
- `uv run ethica run <config> --dry-run` — grid size + wall-time estimate; `--phase generate|score|judge|report` runs one phase
- `uv run ethica corpus check <framework>` · `ethica lora train --model K --framework F` · `ethica label export|import|agreement`

## Study v1 (framework trained into the weights; `docs/study-v1/PROPOSAL.md`, plan in `docs/superpowers/plans/`)
- `uv run python scripts/prepare_base_model.py` — convert the base checkpoint into `models/` and set its chat format
- `uv run python scripts/v1_pipeline.py prompts | generate <arm> | revise <arm> | dataset <arm...> | train <arm> --seed N` — settings in `configs/training/v1-pilot.yaml`; `generate`/`train` refuse to start while another MLX job runs
- Evaluate adapters through `trained_arms:` in an experiment config (method `none`, no framework text); see `configs/experiments/v1-feasibility.yaml`
- `uv run python scripts/v1_gate.py <config> --reference generic` — internalisation gate report from stored scores (needs the `open_dilemmas` benchmark and the judge); `scripts/v1_pool.py --baseline generic <config per training seed>...` — contrasts pooled over training seeds (`docs/study-v1/GATE.md`)
- Nothing trained is ever committed or published: `adapters/`, `models/`, `data/v1/`, `runs/`, `labels/` are gitignored and `tests/test_repo_hygiene.py` enforces it
- Ethics arms need owner-approved `configs/training/commitments/<arm>.md` before generation (D-64)

## Contracts (change deliberately, update all callers)
- `src/ethica/types.py` — Message, ChatBackend, ModelSpec, FrameworkSpec, Variant, EnvEvent, EpisodeResult, Score
- `src/ethica/sandbox/spec.py` — WorldSpec / tripwires / simulated commands
- `src/ethica/scenarios/schema.py` — scenario YAML schema
- `docs/METRICS.md` — every `Score.metric` must be listed there

## Pipeline modules
- `src/ethica/config.py` — ExperimentConfig (YAML); `load_config` = shape only, `validate_refs` = names resolve (run time)
- `src/ethica/orchestrator.py` — `expand_grid`, `generate`, `score`, `judge`, `report`, `estimate`; one model resident at a time
- `src/ethica/training/` — v1: `prompts.py` (fixed prompt set), `generate.py` (teacher answers, screening, resume), `revise.py` (optional critique and revise against the commitments), `dataset.py` (chat-format files), `toolformat.py` (tool-protocol examples shared by all arms), `chat_format.py` (template and tokenizer edits for a base checkpoint)
- `src/ethica/cli.py` — typer app; tests inject backends via `orchestrator.generate(..., backend_factory=)`
- Test helpers: `tests/orch_helpers.py` (mock registry, scripted backend, resident-instance tracker), `tests/fixtures/corpora/` (synthetic corpora, tracked)

## Conventions
- Python 3.12, dataclasses for internal records, pydantic for anything loaded from YAML/JSON/HTTP.
- Comments only when load-bearing (why, not what).
- Tests never load real models unless marked `@pytest.mark.slow`; use `backends.mock`.
- Sandbox tools must never touch the host filesystem/network. Docker executor: `--network none`, read-only, capped.
- `corpora/*` content is gitignored (copyright); only `corpora/_template/` is tracked.
- Hardware target: 16 GB M5. Only one model resident at a time.
