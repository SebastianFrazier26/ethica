# First real signal: RAG framework conditioning on the abliterated model (2026-11-18)

Config `configs/experiments/abl-rag-frameworks.yaml`. qwen3-4b-abl (Instruct-2507,
"gabliterated"), RAG (BM25, padded to 1000 tokens), 9 frameworks × 3 seeds × (40 HarmBench +
40 XSTest-unsafe + 25 XSTest-safe + 48 fidelity). 4131 episodes, 0 errors, 2835 judged.
Report: `runs/abl-rag-frameworks/report/report.md`.

**Status: exploratory.** Corpora are SEP + PD primaries, not yet owner-reviewed; judge not
validated against human labels; one model; benchmark proxies, not agentic scenarios.

## Harmful compliance vs placebo (judge; paired on task × seed, n≈240 pairs, 177 tasks)

| Framework | rate | placebo | risk diff [95% CI] | Holm p | own-framework fidelity |
|---|---|---|---|---|---|
| virtue | 0.28 | 0.46 | **−0.18** [−0.24, −0.11] | **0.0007** | 0.58 |
| deontology | 0.33 | 0.46 | **−0.13** [−0.19, −0.06] | **0.005** | 0.61 |
| care | 0.34 | 0.46 | **−0.11** [−0.18, −0.04] | **0.02** | 0.60 |
| divine command | 0.35 | 0.46 | **−0.11** [−0.19, −0.04] | **0.02** | 0.50 |
| contractualism | 0.35 | 0.46 | **−0.11** [−0.18, −0.04] | **0.02** | 0.48 |
| rule utilitarian | 0.45 | 0.46 | −0.01 [−0.08, +0.06] | 1.0 | 0.50 |
| act utilitarian | 0.48 | 0.46 | +0.03 [−0.05, +0.10] | 1.0 | 0.82 |
| none (no conditioning) | ~0.59 | | | | |

CI = task-clustered bootstrap; p = sign-flip permutation, Holm across the 7 frameworks.

Over-refusal on safe XSTest prompts stays ≤0.05 in every arm — the reductions are not a
blanket "refuse everything" effect.

## Reading

1. **Five non-consequentialist frameworks cut harmful compliance by 11–18 points beyond the
   placebo; both utilitarian frameworks do nothing.** Clean split along the
   consequentialist / non-consequentialist line. Virtue ethics is strongest.
2. Act-utilitarian conditioning is the most *recognisable* (fidelity 0.82) yet the least
   protective — conditioning works, the framework's content just doesn't counsel refusal.
3. The placebo itself lowers compliance (0.59 → 0.46): text in context matters regardless of
   content. Comparing against `none` would have overstated every effect.
4. Caveat: an abliterated model "refusing more" under a framework is the phenomenon under
   study (framework content), not a D-56 violation — D-56 concerns the procedure, which the
   placebo controls for.

## Replication (2026-11-19): qwen3-8b-abl, granite-4.1-3b-abl

Config `configs/experiments/abl-rag-replicate.yaml`: same RAG setup, 2 seeds, no fidelity
items. 3420 episodes, 0 errors, all judged. Report: `runs/abl-rag-replicate/report/report.md`.
Risk difference vs placebo [95% CI], Holm p in brackets:

| Framework | qwen3-4b-abl (3 seeds) | qwen3-8b-abl | granite-4.1-3b-abl |
|---|---|---|---|
| virtue | **−0.18** (0.001) | **−0.20** [−0.28,−0.11] (0.001) | **−0.10** [−0.16,−0.03] (0.047) |
| deontology | **−0.13** (0.005) | **−0.18** [−0.26,−0.09] (0.001) | −0.11 [−0.18,−0.03] (0.068) |
| care | **−0.11** (0.02) | **−0.14** [−0.20,−0.07] (0.001) | −0.06 [−0.13,+0.01] (0.39) |
| contractualism | **−0.11** (0.02) | **−0.14** [−0.22,−0.06] (0.004) | −0.08 [−0.14,−0.01] (0.26) |
| divine command | **−0.11** (0.02) | −0.09 [−0.17,−0.02] (0.08) | **−0.14** [−0.22,−0.07] (0.004) |
| rule utilitarian | −0.01 (1.0) | −0.07 [−0.14,+0.01] (0.20) | −0.01 [−0.09,+0.06] (1.0) |
| act utilitarian | +0.03 (1.0) | −0.06 [−0.15,+0.03] (0.21) | +0.01 [−0.05,+0.08] (1.0) |
| placebo rate (none) | 0.46 (0.59) | 0.60 (0.64) | 0.62 (0.64) |

**It replicates in direction.** All 15 non-consequentialist contrasts across 3 models are
negative (10/15 significant after Holm); all 6 utilitarian contrasts sit in [−0.07, +0.03],
none significant. Over-refusal stays ≤0.04. Effect is largest on Qwen3-8B, smallest on Granite;
which non-consequentialist framework is strongest varies by model (virtue on Qwen, divine
command on Granite). The placebo-vs-none drop is much smaller on 8B/Granite (−0.04/−0.02)
than on 4B (−0.13).

## Threats to validity / next steps

- **Corpus confound:** SEP care/contractualism entries discuss harm and rights explicitly;
  utilitarian entries discuss aggregation. Effect may track topic words, not "framework".
  Owner corpus review + a keyword-matched control would help.
- ~~Single model~~ — replicated on 2 more (above). Still all "strong-abliteration" models;
  untested on the weak pairs and on stock models' agentic behaviour.
- **Judge unvalidated.** Label ~50 items (`ethica label export`) and check agreement.
- **Benchmarks ≠ misalignment.** The agentic scenarios (owner writing) are the actual target;
  builtin ones floor at 0.
