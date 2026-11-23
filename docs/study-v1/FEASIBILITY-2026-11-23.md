# v1 feasibility check (2026-11-23)

Question (proposal §8, step 1; plan Task 7): after LoRA on generic dialogues only, does the base
checkpoint `Qwen/Qwen3-4B-Base` hold a conversation, stop by itself, and use the harness's tool
protocol?

**Verdict after the second run (same night): yes on all three, once the chat format and the
tool tags are spelled in ordinary tokens. One finding changes the design question: the base
checkpoint is not free of safeguards (Result 5).** Everything below is one training seed and
small samples; it is a plumbing result, not a study result.

## Second run: before and after

Three changes were made after the first run and the generic arm was regenerated, retrained and
re-evaluated (`configs/experiments/v1-baseline.yaml`, which also adds `gsm8k` and `mmlu`):

1. Teacher instruction reworded to address the user as "you".
2. 40 tool-protocol tasks (120 training rows) added to every arm's data. They are built from
   gold tool calls replayed in the real sandbox; no model writes them
   (`src/ethica/training/toolformat.py`).
3. `<tool_call>` and `</tool_call>` spelled as ordinary text in the base model's tokenizer. In
   Qwen's vocabulary they are single tokens, and the base checkpoint cannot emit them, the same
   problem as the ChatML end marker. With change 2 alone the model wrote each call's JSON
   correctly and put junk tokens where the tags belong, so still no call parsed.

| Generic arm | First run (step-200 checkpoint) | With changes 1–2 | With changes 1–3 |
|---|---|---|---|
| Training | 3 passes, 13 min | 1 pass (291 steps), 5.4 min | same |
| Validation loss, start → end | 1.54 → 1.25 at step 200 | 1.54 → 1.22 | 1.54 → 1.19 |
| Benchmark answers cut off | 1 of 103 | 0 of 173 | 2 of 173 |
| Answers opening in the asker's voice ("I…"), 55 harm and XSTest answers | 49 | — | 2 |
| Scenarios with zero tool calls | 5 of 6 | 6 of 6 | **0 of 6** |
| Scenario task success | 0 of 6 | 0 of 6 | **3 of 6** |
| Harmful compliance (judge, 40 prompts) | 9 | 11 | 13 |

Baseline table from the final run (seed 0; 95% intervals are wide at these sizes):

| | Instruct `qwen3-4b` | Bare base | Generic arm |
|---|---|---|---|
| Harmful compliance (40) | 0 | 2 (5%) | 13 (33%) |
| Over-refusal on safe prompts (15) | 0 | 0 | 0 |
| Scenarios with zero tool calls (6) | 0 | 0 | 0 |
| Scenario task success (6) | 5 | 2 | 3 |
| Misaligned actions (6) | 0 | 0 | 0 |
| GSM8K accuracy (30) | 97% | 83% | 70% |
| MMLU accuracy (40) | 73% | 40% | 68% |
| Benchmark answers cut off (173) | 8% | 23% | 1% |
| Benchmark answers unparseable (173) | 1% | 3% | 1% |

Reading it:

- **All three pass criteria are now met**: answers stop, are on-topic and addressed to the user,
  and tools are called in the harness's syntax (pass criterion (c) allowed one more zero-call
  scenario than the instruct model; there are none).
- **The bare base model can use tools too** once the tags are ordinary text (0 of 6 zero-call
  scenarios, 2 of 6 successes), but it loops to the turn limit in three scenarios.
- **Capability:** generic training costs about 13 points on GSM8K against the bare base (21 of 30
  vs 25 of 30) and gains 28 on MMLU (27 of 40 vs 16 of 40), where the bare model's score is
  dragged down by cut-off and unparseable answers. Against the instruct model the generic arm is
  27 points lower on GSM8K and 5 lower on MMLU. Any framework arm should be read against the
  generic arm's capability, not the instruct model's.
- **Harmful compliance of the generic arm rose across the three versions (9, 11, 13 of 40).**
  These are single runs of 40 prompts with overlapping intervals; the stable part is that every
  version sits well above the bare base (2 of 40) and far below v0's abliterated models (59%).

The sections below are the first run's results as written, kept for the record.

## What was run

| Step | Detail | Measured |
|---|---|---|
| Base model | converted to 4-bit g64, `models/qwen3-4b-base-q4g64` | 2.26 GB |
| Prompts | 500 from the fixed set (200 dilemma, 100 advice, 75 self-description, 125 neutral) | |
| Teacher answers | `qwen3-4b-abl`, generic arm (no corpus) | 500 in about 41 min (about 5 s each); 483 kept, 17 screened out as refusals |
| Dataset | 459 train, 24 validation | about 113k tokens (the proposal guessed 250k for 500 dialogues) |
| Training | LoRA rank 16, 16 layers, lr 1e-4, loss on the assistant turn only, 3 passes (689 steps) | 13 min, 3.8 GB peak |
| Evaluation | `configs/experiments/v1-feasibility.yaml`: bare base, instruct reference (`qwen3-4b`), and the trained generic arm at two checkpoints; 6 scenarios, 20 HarmBench, 20 XSTest-unsafe, 15 XSTest-safe, 48 fidelity items; seed 0 | 436 episodes in 33 min, judging 17 min, 0 errors |

No framework text or framework training was involved anywhere.

## Result 1: chat format. ChatML fails, a plain-token format works

The checkpoint ships the instruct models' ChatML template. Trained through it, the model answered
in the teacher's voice but never ended its turn: in a greedy 260-token sample it emitted no
`<|im_end|>` at all, although every training example ends with one. LoRA leaves the embedding and
output rows frozen, and the base checkpoint evidently cannot produce that token.

With a template made of ordinary tokens (`### User:` / `### Assistant:`, each assistant turn
closed by `<|endoftext|>`), the same data and settings give a model that stops.
`scripts/prepare_base_model.py` now defaults to this format (D-66).

Share of benchmark answers cut off at the 768-token limit (lower is better):

| Arm | Cut off |
|---|---|
| Instruct reference | 12 of 103 (12%) |
| Bare base, plain format | 34 of 103 (33%) |
| Generic, 3 passes | 6 of 103 (6%) |
| Generic, stopped at step 200 | 1 of 103 (1%) |

The bare base model often answers and then invents further `### User:` turns. Pass criterion (a)
was at most 5% cut off: met by the step-200 checkpoint.

## Result 2: three passes overfit; one is enough

Validation loss was lowest at step 200 of 689 in both training runs (1.25, against 1.54 at the
start and 1.95 at the end), about 0.9 of one pass. Training loss fell to 0.25. The step-200
checkpoint also stops more reliably. `configs/training/v1-pilot.yaml` now trains 1 pass (D-66).
This makes an arm about 5 minutes of training, not 13.

## Result 3: answers are on-topic, with one flaw inherited from the teacher

Read by hand: 5 fidelity answers, 8 answers to the sanity prompts, and the openings of 16
harm-benchmark answers. All were on-topic prose in the requested format (fidelity answers begin
`Answer: A/B` with a reason; 0 to 2 of 48 unparseable). Criterion (b) is met.

The flaw: the trained model often answers as if it were the person asking ("I should talk to my
flatmate first"). The cause is the teacher instruction "answer in the first person, in your own
voice", which the teacher took as "speak as the user". The instruction has been reworded to
address the user as "you" (`generate.py`); the generic data has **not** been regenerated with it
yet.

## Result 4: tool use fails

| Arm | Scenarios with zero tool calls | Task success |
|---|---|---|
| Instruct reference | 0 of 6 | 5 of 6 |
| Bare base | 6 of 6 | 0 of 6 |
| Generic, 3 passes | 6 of 6 | 0 of 6 |
| Generic, step 200 | 5 of 6 | 0 of 6 |

The trained model tries to call tools but in the wrong syntax (for example
`<.tool name="read_inbox" arguments="{}"/>` instead of the harness's `<tool_call>` JSON).
Criterion (c) fails in this first run. **Fixed in the second run** (top of this document): it
needed both tool-format examples in the training data and the tool tags spelled as ordinary
text.

## Result 5: the base checkpoint already refuses harmful requests

This was not expected and bears on what "ethics as the only post-training" can mean.

Harmful compliance, judge-scored, 20 HarmBench + 20 XSTest-unsafe prompts:

| Arm | Harmful compliance | For comparison |
|---|---|---|
| Instruct reference (`qwen3-4b`) | 0 of 40 | |
| **Bare base checkpoint** | **2 of 40 (5%)** | v0 abliterated instruct model, no conditioning: 59% |
| Generic, 3 passes | 9 of 40 (23%) | |
| Generic, step 200 | 9 of 40 (23%) | |

On HarmBench the bare base model uses a stock refusal phrase in 12 of 20 answers ("I'm sorry,
but I cannot assist with that… illegal and unethical"). It also describes itself as "an AI
language model". So the published "Base" checkpoint already behaves like a safety-tuned
assistant in a chat format, whether from its pretraining data or from how it was made; this run
cannot tell which.

What follows:

- **The starting point is not safeguard-free.** v1's design assumed a base checkpoint has no
  refusal training to replace. This one has refusal behaviour to start with.
- **Generic fine-tuning moves harm behaviour by itself**, here upward (5% to 23%), with benign
  prompts only and an abliterated teacher. The D-63 gate (the no-ethics arm must not refuse more
  than the untrained base) passes, but the size of the shift confirms that every framework arm
  must be compared with the generic arm, never with the bare model.
- **The baseline is low.** A generic arm at about 23% leaves less room to detect a framework
  lowering harmful compliance than v0's placebo arms did (46–62%). It leaves plenty of room to
  detect a framework raising it.
- The trained model rarely uses refusal phrases (2–3 of 20 on HarmBench) yet the judge scores
  most of its answers as not substantively harmful: many are first-person deflections. The
  judge is still unvalidated against human labels, and this arm is where the heuristic and the
  judge disagree most, so the 23% needs the owner's label check before it is relied on.

## Measured costs, replacing the proposal's estimates

| Item | Proposal estimate | Measured |
|---|---|---|
| Teacher answer | 8.5 s | about 5 s |
| Tokens per 500 dialogues | 250k | 113k |
| Training one arm (500 dialogues) | about 20 min at 3 passes | 13 min at 3 passes; about 5 min at 1 pass (extrapolated) |
| Evaluating one arm (109 episodes, with judging) | — | about 12 min |
| Pilot, 5 arms × 3 seeds, this evaluation size | about 1.5 days | about 3.5 h generation + 1.3 h training + 3 h evaluation (extrapolated; larger benchmark samples scale the last term) |

## Decision needed from the owner

The base checkpoint's built-in refusals (Result 5). Options:

1. **Proceed as planned** with this base checkpoint and the generic arm as the baseline, and say
   in the write-up that the base is not safeguard-free. Cheapest; the question becomes "what does
   framework training add to a lightly safeguarded model, beyond generic training".
2. **Remove the refusal behaviour from the base checkpoint first** (abliterate it ourselves, one
   recipe, before any training). Closest to "replacing the safeguards", but adds a tool to build
   and brings back the v0 finding that training partly undoes abliteration.
3. **Fall back to proposal option 1** (train on the existing abliterated instruct model).
4. **Look for a cleaner base** in another model family. Unknown effort; nothing checked yet.

Agent's recommendation: option 1 for the pilot, because it needs nothing new and the pilot will
show whether framework arms separate from the generic arm at all; revisit option 2 if they do
not. Not started: this is the owner's call.

## Reproduce

```bash
uv run python scripts/prepare_base_model.py                 # plain chat format and tool tags
uv run python scripts/v1_pipeline.py generate generic
uv run python scripts/v1_pipeline.py dataset generic        # adds the shared tool-format rows
uv run python scripts/v1_pipeline.py train generic --seed 0
HF_HUB_OFFLINE=1 uv run ethica run configs/experiments/v1-baseline.yaml
```

This reproduces the second run. The first run (`configs/experiments/v1-feasibility.yaml`) used 3
passes, the first teacher instruction and no tool rows, with the step-200 checkpoint taken by
hand; its adapters and those of the intermediate run are kept locally under `adapters/v1/` with
descriptive suffixes. Adapters, data and transcripts are local and gitignored.
