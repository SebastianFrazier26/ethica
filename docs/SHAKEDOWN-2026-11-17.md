# Shakedown runs (2026-11-17)

Purpose: first end-to-end runs of the conditioning paths on real corpora. Plumbing, timing and
base rates only — not science (only the 6 builtin scenarios, unreviewed corpora, judge not
validated against human labels).

| Config | What | Status |
|---|---|---|
| `shakedown-lora.yaml` | qwen3-4b × deontology × lora (300 iters, lr 1e-4) | done |
| `shakedown-lora2.yaml` | same, 100 iters, lr 5e-5, plus `lora+prompt` | running |
| `shakedown.yaml` | qwen3-4b pair × {none, placebo, deontology, act_util} × {prompt, rag} | 477/2064 when killed by Claude Code's low-memory reaper (other sessions' jobs); resumed after lora2 |

## shakedown-lora (run 1)

- Training: 467 s wall, 4.0 GB peak, train loss 2.91 → 1.7–1.9, **val loss 3.32 → 2.90 (step
  100) → 2.98 (step 300)**: overfit after ~step 100 (≈3.5 epochs over the 174k-token corpus).
- **Tool protocol broke.** All 6 scenarios ended after 1 turn with `stop_reason=no_tool_call`.
  The model writes `{ "action": "set_config", ... }` as prose without `<tool_call>` tags and
  hallucinates results. Stock qwen3-4b in baseline-4b averages 4–6 turns with real tool calls.
  → LoRA on raw prose degrades instruction/format following; LoRA arms are uninterpretable
  until fixed (task failure ≠ misalignment signal).
- Fidelity probe under the deontology adapter: deontology items 0.62 vs virtue/contractualism
  0.77 — no visible shift toward deontology (no unconditioned comparison yet; shakedown
  `none` arm gives it).
- Judge: harmful_compliance 0.00 (n=50), over_refusal 0.04 (n=25).

## Candidate LoRA fixes (cheapest first)

1. Fewer iters / lower lr (stop near val minimum) — being tested in lora2.
2. `lora+prompt` (framework framing + excerpts in the prompt on top of the adapter) — in lora2.
3. Mix chat-format instruction data (incl. tool-protocol examples) into training so format is
   preserved. Most robust; needs a data decision (which instruction set, licence).
4. Lower `scale` (20 → ~8) to weaken the adapter delta.

## Results: lora2

PENDING.

## Results: shakedown (prompt / rag)

PENDING.
