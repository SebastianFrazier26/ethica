# v1 pilot results (2026-11-25), PROVISIONAL

**All harm, refusal and identification numbers below come from the AI judge** (`judge-default`),
with the interim agent rater as a first-pass check (`docs/JUDGE.md`). They are not validated
against human labels. Owner, 2026-11-24: the judge does the first pass, the owner reviews as
results come in. Review sheets are in `labels/` (local, gitignored):
`v1-pilot-s0-harm-80.csv` (blind; judge verdicts in the `.key.csv`) and
`v1-pilot-s0-identification-60.csv` (answer and judge label side by side; arm in the `.key.csv`).

Setup: five arms trained on `qwen3-4b-base-abl` (ablated base, D-67/D-71), three training seeds
each, second-generation data (D-72); evaluated with no framework text in context, one sampling
seed. Configs `configs/experiments/v1-pilot-s{0,1,2}.yaml`; gate `scripts/v1_gate.py --reference
generic`; pooled contrasts `scripts/v1_pool.py --baseline generic` (training runs resampled per
arm, tasks paired). Run outputs are local under `runs/v1-pilot-s*/` and `runs/v1-pilot-pooled/`.

## 1. Internalisation gate: did the framework get in?

Identification on 24 held-out open dilemmas × 2 phrasings (48 answers per arm and seed). Pass
rule (owner-accepted, D-68): the case-clustered 95% interval lies above chance (1/8) and above
the untrained-or-generic reference arm's rate for the same label.

| Arm | Seed 0 | Seed 1 | Seed 2 | Generic arm gets this label | Same label both phrasings (mean) |
|---|---|---|---|---|---|
| Act utilitarian | 89% [79, 97] pass | 97% [93, 100] pass | 100% [100, 100] pass | 52% | 94% |
| Deontology | 66% [52, 79] pass | 50% [35, 64] pass | 85% [75, 93] pass | 1% | 52% |
| Virtue | 27% [14, 39] pass | 47% [33, 62] pass | 18% [10, 29] **fail** | 0% | 25% |

Mean label shares over the three seeds:

| Arm | Act util. | Care | Contract. | Deontology | None | Rule util. | Virtue |
|---|---|---|---|---|---|---|---|
| Act utilitarian | **96%** | 0 | 0 | 0 | 2% | 2% | 0 |
| Deontology | 8% | 18% | 3% | **67%** | 3% | 1% | 0 |
| Virtue | 18% | **35%** | 0 | 1% | 13% | 1% | 31% |
| Generic (no ethics) | 52% | 20% | 3% | 1% | 19% | 6% | 0 |
| Placebo | 48% | 19% | 1% | 0 | 24% | 7% | 1% |
| Untrained ablated base | 49% | 11% | 4% | 1% | 30% | 6% | 0 |

Reading it:

- **Deontology got in**, in all three seeds, from a 1% starting rate.
- **Act utilitarianism got in, but the bar is high.** Untrained and generic models already read
  as act-utilitarian about half the time. The trained arm reaches 89–100%, and the gap is clear.
- **Virtue got in weakly.** It passes in 2 of 3 seeds. The judge labels its answers as care
  ethics more often than as virtue. Either the training signal is weak, or the judge can't tell
  the two apart. The label review sheet can separate these.
- **Persistence** (a contrary framework's corpus in the system prompt) barely moves any arm.
  Act utilitarian under deontology pressure: 97% still act-utilitarian. Deontology under
  act-utilitarian pressure: 63% deontology and 21% act-utilitarian, against 8% unpressured.
  Virtue under act-utilitarian pressure: own label 40%, act-utilitarian 37%. The trained outlook
  mostly survives an in-context push.
- Phrasing consistency of the controls is 33–58%. Act utilitarian is at 94%. Deontology (52%) and
  virtue (25%) are no more stable than the controls.

## 2. Harmful compliance (40 HarmBench + 40 XSTest-unsafe; judge)

| Arm | Seed 0 | Seed 1 | Seed 2 | Pooled difference vs generic [95% CI] | Same side in every run pair |
|---|---|---|---|---|---|
| Generic (reference) | 59% | 54% | 44% | — | |
| Untrained ablated base | 33% | 33% | 33% | −0.20 [−0.33, −0.08] | yes |
| Placebo | 54% | 37% | 51% | −0.04 [−0.18, +0.10] | no |
| **Deontology** | 49% | 28% | 19% | **−0.20 [−0.38, −0.02]** | no |
| **Virtue** | 39% | 33% | 46% | **−0.14 [−0.26, −0.00]** | no |
| Act utilitarian | 45% | 59% | 57% | +0.03 [−0.10, +0.17] | no |

Over-refusal on 20 XSTest-safe prompts: 0–10% in every run, with no pooled difference. Judged
overt refusal is 0 in every trained arm.

Reading it, with the caveats that follow:

- The v0 ordering shows up again with the framework in the weights. Deontology and virtue lower
  harmful compliance relative to the generic arm; act utilitarianism does not. The placebo
  doesn't either, so topic or register of training text alone doesn't explain it.
- **Neither result is robust.** Both intervals end just short of zero, and no ethics arm is below
  the generic arm in every pairing of runs. Deontology's spread across seeds (49%, 28%, 19%) is
  larger than its pooled effect.
- **No trained arm refuses.** The phrase heuristic finds a refusal phrase in at most 4% of
  trained-arm harm answers. The differences are entirely the judge's call on whether an answer
  gives substantive harmful help. This is the result most in need of the owner's label check.
- Training without ethics raises compliance (base 33% → generic 44–59%), as on the unablated base.

## 3. Capability

| Arm | GSM8K (30), seeds 0/1/2 | MMLU (40), seeds 0/1/2 | Benchmark answers unparseable, seed 2 |
|---|---|---|---|
| Untrained ablated base | 77% | 47% | 3% |
| Generic | 63 / 57 / 60% | 62 / 57 / 50% | 2% |
| Placebo | 57 / 37 / 47% | 47 / 70 / 30% | 24% |
| Virtue | 70 / 50 / 40% | 53 / 65 / 68% | 0% |
| Deontology | 70 / 23 / 20% | 50 / 53 / **17%** | 19% |
| Act utilitarian | 70 / 37 / 27% | 50 / 65 / 33% | 2% |

- **Q4 (GSM8K drop):** the generic arm's 30% in the first-data re-alignment check does not
  recur. With the D-72 data, generic is at 57–63% on all three seeds. The pooled capability
  differences against generic all include zero.
- **The ethics and placebo arms are far less stable across seeds than generic.** Seed 0 of every
  arm is near 70% on GSM8K; seeds 1 and 2 of the framework arms fall to 20–50%.
- **Deontology seed 2 lost its answer format** (19% unparseable, MMLU below chance). It is also
  the run with the lowest harmful compliance (19%). Part of deontology's harm reduction may come
  from a weaker model, not from the framework. The same pattern does not hold for every run:
  seed 1 has normal MMLU (53%) and low compliance (28%), but low GSM8K (23%). A reading that
  controls for capability is needed before deontology's harm effect is claimed.
- Scenarios: no misaligned action in any run. Task success is 0–4 of 6 per run, with no pooled
  difference.

## What this supports and what it doesn't (provisional)

Supported, given the judge: LoRA on about 445 synthetic first-person dialogues puts act
utilitarianism and deontology into the base model's unprompted moral reasoning, strongly enough
to survive a contrary framework in context. Virtue gets in weakly.

Not yet supported: that a trained framework changes harmful compliance. The direction matches
v0, but the intervals barely exclude zero, refusals are absent, and the arm with the largest
effect is also the one whose capability varies most across seeds.

## Next steps (owner calls)

1. Owner review of the two label sheets, starting with harm, since every harm difference rests on
   the judge.
2. A capability-adjusted harm contrast (harm per run against that run's MMLU/GSM8K), or more
   training seeds, to separate the deontology effect from run quality.
3. Whether virtue's training signal needs strengthening (critique and revise, D-70) or the judge
   needs a better virtue/care distinction.
