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

## Contracts (change deliberately, update all callers)
- `src/ethica/types.py` — Message, ChatBackend, ModelSpec, FrameworkSpec, Variant, EnvEvent, EpisodeResult, Score
- `src/ethica/sandbox/spec.py` — WorldSpec / tripwires / simulated commands
- `src/ethica/scenarios/schema.py` — scenario YAML schema
- `docs/METRICS.md` — every `Score.metric` must be listed there

## Pipeline modules
- `src/ethica/config.py` — ExperimentConfig (YAML); `load_config` = shape only, `validate_refs` = names resolve (run time)
- `src/ethica/orchestrator.py` — `expand_grid`, `generate`, `score`, `judge`, `report`, `estimate`; one model resident at a time
- `src/ethica/cli.py` — typer app; tests inject backends via `orchestrator.generate(..., backend_factory=)`
- Test helpers: `tests/orch_helpers.py` (mock registry, scripted backend, resident-instance tracker), `tests/fixtures/corpora/` (synthetic corpora, tracked)

## Conventions
- Python 3.12, dataclasses for internal records, pydantic for anything loaded from YAML/JSON/HTTP.
- Comments only when load-bearing (why, not what).
- Tests never load real models unless marked `@pytest.mark.slow`; use `backends.mock`.
- Sandbox tools must never touch the host filesystem/network. Docker executor: `--network none`, read-only, capped.
- `corpora/*` content is gitignored (copyright); only `corpora/_template/` is tracked.
- Hardware target: 16 GB M5. Only one model resident at a time.
