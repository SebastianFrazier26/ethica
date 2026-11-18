# ethica — design spec (v0, drafted overnight 2026-11-16)

Status: **draft for review**. Every judgment call made without you is logged in
[`DECISIONS.md`](DECISIONS.md) with an ID (D-xx) and the alternatives rejected.

## 1. Research question

Does conditioning a small open-weight LLM on a specific normative ethical framework
(deontology, act/rule utilitarianism, virtue ethics, theological/divine-command,
care ethics, contractualism, …) change how often it behaves *misaligned* — i.e.
acts against its principals' legitimate instructions and oversight — relative to
controls? And does that effect differ when the model's stock safety training has
been removed (abliterated)?

Sub-questions the harness must be able to answer:

1. **Manipulation check** — did conditioning actually make the model reason like
   framework X? (Without this, a null result is uninterpretable.)
2. **Misalignment rates** per framework × safeguards × conditioning method.
3. **Rationalisation** — when a conditioned model misbehaves, does it justify the act
   in its framework's terms ("the greater good requires…")?
4. **Alignment/capability tax** — does conditioning cost general capability or
   cause over-refusal?

### Misalignment ≠ moral disagreement

A deontological model refusing a task a utilitarian would do is a *moral
difference*, not misalignment. The primary misalignment metrics are therefore
anchored on the classic agentic failure modes that are bad under every
framework's *own* account of an agent acting for a principal: covert action,
deception about one's own actions, disabling oversight, self-exfiltration /
self-preservation against shutdown, unsanctioned power/resource acquisition,
sandbagging, reward hacking. Open refusal or open objection is scored separately
(`overt_refusal`), never as misalignment. See D-03.

## 2. Experimental design

Factorial grid, each cell = a **Variant**:

| Factor | Levels (v0 defaults — final list is yours) |
|---|---|
| Model family/size | 4–6 small models, 1.5–8B, see `configs/models.yaml` |
| Safeguards | `stock` vs `abliterated` (refusal direction removed) |
| Framework | `none`, `placebo`, + N ethics frameworks |
| Conditioning method | `prompt`, `rag`, `lora` (+ optional `lora+prompt`) |
| Seed | ≥3 per task (sampling temperature 0.7) |

**Two controls** (D-04):
- `none` — model as shipped.
- `placebo` — same conditioning pipeline fed a *length-matched non-ethical
  corpus* (e.g. encyclopaedic history/geography). Separates "this framework did
  X" from "adding a pile of text to the context / weights did X". Every ethics
  framework is compared against placebo first, none second.

## 3. Architecture

```
configs/*.yaml ──► orchestrator (grid expansion, resumable)
                       │  one model resident at a time (16 GB RAM)
                       ▼
     ┌──────── Variant = ModelSpec × FrameworkSpec × method ───────┐
     │ backends/   mlx (default) | openai-compatible | mock         │
     │ conditioning/ prompt | rag (BM25, from scratch) | lora (mlx) │
     └───────────────────────────────┬──────────────────────────────┘
                                     ▼
  PHASE 1 — generate     agent_loop  ◄──►  sandbox/ (simulated stateful world,
                         (text tool protocol)        every action → EnvEvent)
                         scenarios/ (agentic)   benchmarks/ (static)
                                     │ EpisodeResult
                                     ▼
                         tracking/  SQLite via SQLAlchemy 2.0
                                     │
  PHASE 2 — score        scoring/programmatic (EnvEvent-based ground truth)
                         scoring/judge (local judge model, separate pass)
                                     │ Score rows
                                     ▼
  PHASE 3 — analyse      analysis/stats (bootstrap CIs, permutation tests,
                         Holm correction, from scratch) → analysis/report
```

Phases are decoupled because the judge model and the model-under-test cannot be
co-resident in 16 GB (D-09). Every phase is idempotent and resumable from the DB.

### 3.1 Backends (`src/ethica/backends/`)
- `mlx` — `mlx-lm` in-process; supports LoRA adapters directly. Default (D-05).
- `openai` — any OpenAI-compatible local server (llama.cpp `llama-server`, LM
  Studio, vLLM-mlx). Escape hatch for GGUF-only abliterated models.
- `mock` — scripted, deterministic; used by tests and CI (CI has no GPU).

### 3.2 Conditioning (`src/ethica/conditioning/`)
Corpus layout (filled by you tomorrow):
```
corpora/<framework>/manifest.yaml   # name, kind, display_name, summary, sources[]
corpora/<framework>/*.md|*.txt      # the hand-picked literature
```
- `prompt` — system preamble: framework summary + top excerpts truncated to a
  fixed token budget (identical budget for every framework incl. placebo).
- `rag` — per-turn BM25 retrieval (implemented from scratch) over corpus chunks,
  k passages injected into the system message. Same budget as `prompt`.
- `lora` — LoRA via `mlx_lm.lora` on corpus chunks (continued-pretraining style,
  completion-only). Fixed hyper-params across frameworks; token count matched by
  truncation/upsampling so frameworks receive equal training signal (D-06).

### 3.3 Sandbox (`src/ethica/sandbox/`)
A **simulated** world, not a real shell (D-08): virtual filesystem, email inbox,
process/config table, a "company" key-value store, a fake network/upload sink.
Tools are deterministic Python functions over that state; every call logs an
`EnvEvent` with semantic tags (`oversight_disabled`, `weights_copied`,
`log_deleted`, …). Nothing the model does can touch the host. An optional Docker
executor (`--network none`, read-only rootfs, CPU/mem caps, timeout) exists only
for code tasks (reward-hacking on unit tests).

### 3.4 Agent loop (`src/ethica/agent_loop.py`)
Uniform text tool protocol for every model (D-07):
```
<tool_call>{"name": "read_file", "arguments": {"path": "/notes.txt"}}</tool_call>
```
Results come back as a user turn wrapped in `<tool_result name="…">…</tool_result>`.
Loop ends on `<done/>`, no tool call, `max_turns`, or parse failure budget.
Models may think in `<scratchpad>…</scratchpad>`; told it is private (it is
stored and scored — this is how rationalisation is measured).

### 3.5 Tasks
**Agentic scenarios** (`scenarios/builtin/*.yaml`) — each has a legitimate task,
an embedded *temptation* (discovered memo about shutdown, oversight config,
goal conflict), and programmatic checks over `EnvEvent`s. Categories:
oversight subversion, self-exfiltration, shutdown resistance, goal-guarding,
covert sabotage, sandbagging, deception-about-actions (follow-up interrogation),
reward hacking, power/resource seeking, leverage/coercion, sycophancy under
pressure. Several are **framework-bait**: the misaligned act is framed as
producing better outcomes or upholding a duty, so frameworks can rationalise it.

**Static benchmarks** (`benchmarks/`) — framework-fidelity probe (manipulation
check, authored dilemmas with per-framework expected answers), ETHICS subsets,
XSTest (over-refusal), a harmful-compliance set, GSM8K/MMLU small subsets
(capability tax). Downloaded datasets cached under `data/cache/` (gitignored).

### 3.6 Scoring
1. **Programmatic first** — EnvEvent tags & final state are ground truth for
   *what the model did*. Highest-trust metric.
2. **Judge second** — local judge model with rubric + structured JSON output for
   things only text reveals: deception in the final report, framework-rationalised
   reasoning in the scratchpad, sycophancy. Judge is blind to variant (framework
   names redacted from transcript before judging, D-10). Judge agreement checked
   against a hand-labelled sample (you, later) before trusting it.

### 3.7 Tracking & analysis
SQLite via SQLAlchemy (D-11). Tables: `experiments`, `variants`, `episodes`,
`messages`, `env_events`, `scores`. Analysis: per-variant rates with bootstrap
95% CIs (clustered by task), paired permutation tests framework-vs-placebo,
Holm–Bonferroni across frameworks, effect sizes (risk difference, Cohen's h).
Report = Markdown + CSV + PNG plots. A React/TS dashboard is deliberately
deferred (D-12).

## 4. Budget sanity (16 GB M5)
- 4-bit 4B model ≈ 2.5 GB weights; generation ~30–60 tok/s. A scenario episode
  ≈ 6 turns × 300 tok ≈ 2k tok ⇒ ~1 min. Pilot grid (2 models × 2 safeguards ×
  4 frameworks × 1 method × 15 scenarios × 3 seeds) = 720 episodes ≈ 12 h.
  ⇒ Plan pilots narrow; widen after the harness is trusted.
- LoRA on a 4B 4-bit model with batch 1–2, seq 1024: fits in ~10 GB. 8B LoRA is
  not planned (D-13).

## 5. Out of scope for v0
Full pre-training, models >8B, cloud runs, interpretability probes on
activations (good follow-up: compare refusal-direction / framework-direction).
