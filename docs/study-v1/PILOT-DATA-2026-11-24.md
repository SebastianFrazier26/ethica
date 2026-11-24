# v1 pilot training data (2026-11-24)

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
