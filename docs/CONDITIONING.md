# Conditioning

Code: `src/ethica/conditioning/`. Entry point: `factory.make_conditioner(variant, corpora_root, budget_tokens, k)`.

## Methods

| Method | What changes | Per-framework knob held equal |
|---|---|---|
| `none` | nothing (`NoneConditioner`) | n/a |
| `prompt` | system message gets a `<background_reading>` block: `FRAMING` + summary + evenly spaced corpus excerpts | total block <= `budget_tokens` |
| `rag` | same block, but excerpts are the top-`k` BM25 chunks for the latest user/tool turn (+ last assistant turn); replaced every turn | total block <= `budget_tokens` |
| `lora` | adapter weights; `prepare()` is identity, backend loads `LoRAConditioner.adapter_path` | `target_tokens` of training text, fixed `LoRAConfig` |
| `lora+prompt` | both | both |

`FRAMING` (in `prompt.py`) is one constant used for every framework including `placebo`. It avoids the word
"ethical" so the placebo is not mis-described; the framework enters only via `display_name`, `summary` and excerpts.

## Token budgets

Counts use `approx_tokens` = ceil(chars/4), model-independent so every framework and model gets an identical budget.
Real tokenizers deviate ~+-15%; a measured case (placebo text, Qwen3 tokenizer) was ~6% fewer real tokens than the
estimate. Pass `count=hf_counter(tokenizer)` to the constructors / `build_lora_dataset` to budget in real tokens.
The block budget includes framing, tags and summary, so a longer summary means fewer excerpt tokens; the last
excerpt is truncated so blocks land within a few tokens of the budget.

## Excerpt selection and retrieval

- `prompt`: corpus chunked at 150 tokens; evenly spaced chunks from a seeded phase offset (`seed`), kept in document order.
- `rag`: index chunks of 200 tokens, 40 overlap, never crossing documents. BM25 Okapi in `bm25.py` (k1=1.5, b=0.75,
  Lucene-style non-negative idf, small stopword list). Each retrieved passage is capped at `budget/k` tokens.
  Index persists as JSON (`BM25Index.save/load`; postings are rebuilt on load).

## Adding a framework

1. Copy `corpora/_template/` to `corpora/<name>/` and follow its README (name must equal the directory).
2. Add the entry to `corpora/_frameworks.yaml` (placeholder summaries there are unreviewed).
3. `list_frameworks(root)` picks it up; `make_conditioner` works with no code change.

The `placebo` corpus is built by `placebo.build_placebo_corpus` from Simple English Wikipedia
(`wikimedia/wikipedia`, `20231101.simple`, CC BY-SA 3.0 / GFDL) with a keyword topic/normative filter recorded in its manifest.

## LoRA

Subprocess `python -c <wrapper around mlx_lm.lora.main>` driven by a generated YAML (`lora_config.yaml`; rank is only
settable via YAML in mlx-lm 0.32.2). Dataset: `build_lora_dataset(corpus, dir, target_tokens, seed)` shuffles chunks (512 tokens),
holds out 5% (>=4) for validation, then cycles/truncates the rest to exactly `target_tokens`. mlx-lm trains
`iters x batch_size` samples regardless of file size, so set `target_tokens` near `suggested_target_tokens(cfg)` for ~one epoch.

Defaults (`LoRAConfig`): rank 16, scale 20, dropout 0, 16 layers, batch 2, max_seq 1024, 300 iters, lr 1e-4,
grad checkpointing on, seed 0. Outputs in `out_dir`: `adapters.safetensors`, `adapter_config.json`, `train_meta.json`
(config, data hash, loss curves, peak MLX memory, wall time), `train.log`.

Measured on the M5 16 GB (while other jobs were running, so treat times as upper bounds):

| Model | iters | peak MLX mem | speed |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct-4bit | 5 | 1.24 GB | ~5 s total |
| Qwen3-4B-MLX-4bit | 20 (batch 2, ~365 tok/sample) | 3.74 GB | ~220 tok/s, 79 s incl. load -> ~15-20 min for 300 iters |

Peak memory is for ~365-token samples; samples near max_seq 1024 will use more, but the 10 GB budget in DESIGN s4 has wide headroom.
