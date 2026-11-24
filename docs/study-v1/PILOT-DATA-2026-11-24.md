# v1 pilot training data (2026-11-24)

**Current data: second generation (D-72), section "After the fixes" at the end.** The sections
before it describe the first generation, kept for the record; its adapters are in
`adapters/v1/abl-first-data/` and are not used.

Teacher `qwen3-4b-abl`, 500 pilot prompts per arm, settings in `configs/training/v1-pilot.yaml`.
Ethics arms were generated with the owner-approved commitments (D-64) and their corpus retrieved
into context; placebo with the geology corpus; generic with nothing. Answers and training files
are local (`data/v1/`, gitignored).

## Screening

| Arm | Kept | Screened: refusal phrase | Screened: names the framework | Other |
|---|---|---|---|---|
| generic | 493 | 7 | — | 0 |
| placebo | 497 | 2 | — | 1 (mentions the reading) |
| virtue | 465 | 7 | 28 | 0 |
| deontology | 439 | **31** | 30 | 0 |
| act utilitarian | 481 | 5 | 14 | 0 |

`dataset` keeps only prompts kept in every arm, so each arm trains on the same **400 prompts**
(plus the 120 shared tool-protocol rows): 494 training and 26 validation rows, about 130–144k
tokens per arm.

## Three observed properties, not corrected

**1. The refusal screen falls mostly on deontology.** 26 of its 31 "refusals" match the pattern
"I can't / I won't …", which in a deontological answer is usually the framework speaking ("I
won't lie to her"), not the teacher declining the request. Those prompts are dropped from every
arm (the prompt set is shared), so no arm is unbalanced, but the deontology arm loses some of
its most absolute answers. Most "names the framework" hits are ordinary phrases: "moral law" in
22 of 30 deontology answers, "virtue" in 22 of 28 virtue answers.

**2. Ethics arms often answer as the person asking.** Share of kept dilemma and advice answers
with "I should / would / must / will / need to / have to" in the first 300 characters:

| generic | placebo | virtue | deontology | act utilitarian |
|---|---|---|---|---|
| 1% | 1% | 20% | 44% | 43% |

Read by hand (8 deontology samples), about half are the teacher's own view ("I would start by
asking whether you…") and half answer as the asker ("I should tell the client the delay was
caused by us"). The likely cause is that the commitments are written in the first person and
placed right before the instruction. The feasibility check fixed the same problem in the controls
(D-66c), and found that answering as the asker turns harmful prompts into first-person
deflections that the judge scores as not harmful. This could therefore make ethics arms look
safer for a reason unrelated to ethics.

**3. The placebo's geology drift** (proposal section 2): 123 of 372 kept non-neutral answers use
geology or weather words, against 2 of 368 for the generic arm.

Points 1 and 2 are open questions for the owner. Training went ahead on the data as it is, so
the pipeline and the re-alignment check are not held up; fixing either means rebuilding the
training files (point 1) or regenerating the three ethics arms (point 2, about 7.5 h) and
retraining (about 1.5 h).

## Training (2026-11-24)

All 15 pilot adapters trained on `qwen3-4b-base-abl` (5 arms × seeds 0–2; LoRA rank 16, 16
layers, one pass = 247 steps, about 5 min and 4.4 GB peak each), under `adapters/v1/abl/`
(gitignored, never published). Validation loss (26 rows, so noisy), start → end:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| generic | 1.82 → 1.36 | 1.85 → 1.27 | 1.76 → 1.44 |
| placebo | 2.04 → 1.44 | 1.98 → 1.30 | 1.86 → 1.59 |
| virtue | 2.05 → 1.36 | 2.12 → 1.32 | 2.02 → 1.36 |
| deontology | 1.98 → 1.29 | 1.98 → 1.24 | 1.93 → 1.26 |
| act utilitarian | 2.04 → 1.38 | 2.05 → 1.28 | 1.93 → 1.46 |

In 3 of the 15 runs the loss rose over the last 47 steps (generic s2, placebo s2, act-utilitarian
s2). With one pass fixed by D-66, the final checkpoint is used in every run; no checkpoint was
picked by hand.

Evaluation configs `configs/experiments/v1-pilot-s{0,1,2}.yaml` are written and validated but
**not run**: the pilot is scored only after the owner reviews the open dilemmas and framework
descriptions (D-68). The dry run estimates about 6 h per seed, so about 18 h of machine time for
all three.

## After the fixes (D-72, owner 2026-11-24)

Two changes, then the three ethics arms regenerated, all training files rebuilt and all 15
adapters retrained:

1. Ethics arms get one more sentence after their commitments: advise the user as "you", never
   answer in their place, "I" only for the model's own view (`COMMITMENTS_VOICE`). The controls'
   instruction is unchanged, so their stored answers stand.
2. The training-data refusal screen matches only declining the request ("can't help/assist with
   that", "I must decline", ...), not the benchmark's broader heuristic. Re-applied to the stored
   control answers with `v1_pipeline.py rescreen`: all 52 earlier "refusals" across the five arms
   (generic 7, placebo 2, virtue 7, deontology 31, act utilitarian 5) were false positives, and
   none of the regenerated answers is screened as a refusal.

| Arm | Kept | Screened: refusal | Screened: names the framework | Other | Asker's voice (dilemma and advice, first generation → now) |
|---|---|---|---|---|---|
| generic | 500 | 0 | — | 0 | 1% → 1% (3 of 300) |
| placebo | 499 | 0 | — | 1 meta | 1% → 1% (2 of 300) |
| virtue | 473 | 0 | 27 | 0 | 20% → 0% (1 of 287) |
| deontology | 478 | 0 | 22 | 0 | 44% → 1% (2 of 291) |
| act utilitarian | 480 | 0 | 20 | 0 | 43% → 0% (0 of 289) |

"Asker's voice" is the crude count used above ("I should / would / must / will / need to / have
to" in the first 300 characters). The shared prompt set is now **445 prompts** (from 400), plus
the 120 tool rows: 537 training and 28 validation rows, about 141–154k tokens per arm.
