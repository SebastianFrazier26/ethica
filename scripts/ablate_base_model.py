"""Remove the refusal direction from the v1 base checkpoint (D-67, owner decision 2026-11-23).

    uv run python scripts/ablate_base_model.py

Steps (method in `ethica.training.ablation`):
1. Prompts: AdvBench-derived harmful behaviours and Alpaca harmless instructions, pinned Hub
   revisions, with any harmful prompt close to a HarmBench or XSTest prompt dropped, so the
   direction is never fitted on what the evaluation asks.
2. Residual activations at the last prompt token, every layer, on the 4-bit copy that training
   and evaluation use, in its plain chat format (D-66). The bf16 checkpoint does not fit beside
   the other jobs on a 16 GB machine; the 4-bit copy is also the model the direction is for.
3. Candidate layers: ablate each layer's direction at run time and measure held-out harmful
   refusal, harmless refusal and next-token KL on harmless prompts; pick with `select_layer`.
4. Edit the bf16 weights shard by shard, untie the head, quantize to 4-bit g64 like every other
   model here, set the plain chat format.

Output: models/qwen3-4b-base-abl-q4g64/ + provenance.json with the candidate table (gitignored).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from harmonize_quant import MODELS_DIR, TAG, TARGET, smoke, wait_for_ram

from ethica.benchmarks.base import Source, fetch
from ethica.benchmarks.extract import is_refusal
from ethica.benchmarks.safety import HarmBenchBenchmark, XSTestBenchmark
from ethica.training.ablation import (
    EMBED,
    LM_HEAD,
    Candidate,
    attach,
    drop_near_duplicates,
    kl_divergence,
    mean_diff_directions,
    orthogonalize_embedding,
    orthogonalize_writer,
    select_layer,
    writer_names,
)
from ethica.training.chat_format import apply_chat_format

KEY = "qwen3-4b-base-abl"
BASE_REPO = "Qwen/Qwen3-4B-Base"
BASE_SHA = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
# 4-bit copy with the plain chat format and the tool tags as text, as used in training
FIT_DIR = MODELS_DIR / f"qwen3-4b-base-{TAG}"
HARMFUL = Source(
    "mlabonne/harmful_behaviors",
    "data/train-00000-of-00001.parquet",
    "01cead01398926d81f7c52bdb790ee8cf77ebba7",
    "AdvBench (MIT); used to fit a direction only, never stored or trained on",
)
HARMLESS = Source(
    "mlabonne/harmless_alpaca",
    "data/train-00000-of-00001.parquet",
    "02c6a92cfcf11bb0c387334f8146d149d65b587f",
    "Alpaca (CC-BY-NC-4.0); used to fit a direction only, never stored or trained on",
)


def load_prompts(n_fit: int, n_val: int, seed: int) -> dict[str, list[str]]:
    import pandas as pd

    banned = [i.prompt for i in HarmBenchBenchmark().load(n=10_000)]
    banned += [i.prompt for i in XSTestBenchmark("unsafe", 200).load(n=10_000)]
    harmful_all = pd.read_parquet(fetch(HARMFUL))["text"].str.strip().tolist()
    harmful = drop_near_duplicates(harmful_all, banned)
    harmless = pd.read_parquet(fetch(HARMLESS))["text"].str.strip().tolist()
    rng = random.Random(seed)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    need = n_fit + n_val
    if len(harmful) < need:
        raise SystemExit(f"only {len(harmful)} harmful prompts after the overlap filter")
    return {
        "harmful_fit": harmful[:n_fit],
        "harmful_val": harmful[n_fit:need],
        "harmless_fit": harmless[:n_fit],
        "harmless_val": harmless[n_fit:need],
        "_dropped_as_overlap": [str(len(harmful_all) - len(harmful))],
    }


def chat_tokens(tokenizer: Any, prompt: str) -> list[int]:
    toks = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    return list(toks)


def last_token_residuals(model: Any, tokens: list[int]) -> np.ndarray:
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    h = inner.embed_tokens(mx.array([tokens]))
    mask = create_attention_mask(h, None)
    out = []
    for layer in inner.layers:
        h = layer(h, mask, None)
        out.append(h[0, -1].astype(mx.float32))
    stacked = mx.stack(out)
    mx.eval(stacked)
    return np.array(stacked)


def next_token_logprobs(model: Any, tokens: list[int]) -> np.ndarray:
    import mlx.core as mx

    logits = model(mx.array([tokens]))[0, -1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    mx.eval(lp)
    return np.array(lp)


def refusal_rate(model: Any, tokenizer: Any, prompts: list[str], max_tokens: int) -> float:
    from mlx_lm import generate

    hits = 0
    for p in prompts:
        text = generate(model, tokenizer, chat_tokens(tokenizer, p), max_tokens=max_tokens)
        hits += is_refusal(text)
    return hits / len(prompts)


def fit_and_select(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer

    prompts = load_prompts(args.n_fit, args.n_val, args.seed)
    wait_for_ram(3.5)
    model, _ = load_model(FIT_DIR)
    tok = load_tokenizer(FIT_DIR)
    t0 = time.time()
    acts = {
        k: np.stack([last_token_residuals(model, chat_tokens(tok, p)) for p in prompts[k]])
        for k in ("harmful_fit", "harmless_fit")
    }
    dirs = mean_diff_directions(acts["harmful_fit"], acts["harmless_fit"])
    print(f"activations: {time.time() - t0:.0f} s")
    val_tokens = [chat_tokens(tok, p) for p in prompts["harmless_val"]]
    base_lp = np.stack([next_token_logprobs(model, t) for t in val_tokens])
    base = {
        "harmful_refusal": refusal_rate(model, tok, prompts["harmful_val"], args.max_tokens),
        "harmless_refusal": refusal_rate(model, tok, prompts["harmless_val"], args.max_tokens),
    }
    print("unablated:", base)
    n_layers = len(model.model.layers)
    lo, hi = round(n_layers * 0.25), round(n_layers * 0.8)
    cands = []
    for layer in range(lo, hi, args.layer_step):
        undo = attach(model, mx.array(dirs[layer].astype(np.float32)).astype(mx.bfloat16))
        try:
            lp = np.stack([next_token_logprobs(model, t) for t in val_tokens])
            c = Candidate(
                layer,
                refusal_rate(model, tok, prompts["harmful_val"], args.max_tokens),
                refusal_rate(model, tok, prompts["harmless_val"], args.max_tokens),
                kl_divergence(base_lp, lp),
            )
        finally:
            undo()
        print(c, f"{time.time() - t0:.0f} s", flush=True)
        cands.append(c)
    # relative to the unablated model: the phrase heuristic fires on some harmless answers too
    best = select_layer(cands, max_harmless_refusal=base["harmless_refusal"] + 0.1)
    mu_harm, mu_safe = acts["harmful_fit"].mean(0), acts["harmless_fit"].mean(0)
    del model
    gc.collect()
    mx.clear_cache()
    record = {
        "method": "difference of means at the last prompt token of the 4-bit copy; bf16 weights "
        "orthogonalised (embedding, o_proj, down_proj), head untied and left original, then "
        "quantized to 4-bit g64",
        "fitted_on": str(FIT_DIR.relative_to(MODELS_DIR.parent)),
        "harmful_source": {"repo": HARMFUL.repo, "revision": HARMFUL.revision},
        "harmless_source": {"repo": HARMLESS.repo, "revision": HARMLESS.revision},
        "n_fit": args.n_fit,
        "n_val": args.n_val,
        "harmful_dropped_as_overlap_with_eval": int(prompts["_dropped_as_overlap"][0]),
        "seed": args.seed,
        "max_tokens_for_refusal_check": args.max_tokens,
        "unablated": base,
        "candidates": [c.__dict__ for c in cands],
        "selected_layer": best.layer,
        "diff_to_harmless_norm_ratio_selected": float(
            np.linalg.norm(mu_harm[best.layer] - mu_safe[best.layer])
            / np.linalg.norm(mu_safe[best.layer])
        ),
    }
    return dirs[best.layer], record


def write_ablated(source: Path, r: np.ndarray, out: Path, shard_gb: float = 1.0) -> None:
    # tensor by tensor into ~1 GB shards: a 4 GB source shard, edited in memory, does not fit
    # beside the other jobs on this machine
    import mlx.core as mx

    cfg = json.loads((source / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    writers = set(writer_names(n_layers))
    out.mkdir(parents=True)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = {}
    buf: dict[str, Any] = {}
    size = 0
    n_shard = 0
    edited = 0

    def flush() -> None:
        nonlocal buf, size, n_shard
        if not buf:
            return
        n_shard += 1
        name = f"model-{n_shard:05d}.safetensors"
        mx.save_safetensors(str(out / name), buf, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(buf, name))
        buf, size = {}, 0
        gc.collect()

    def put(k: str, v: Any) -> None:
        nonlocal size
        buf[k] = v
        size += v.nbytes
        if size > shard_gb * 1e9:
            flush()

    for shard in sorted({*index["weight_map"].values()}):
        # mx.load is lazy: each tensor is read when evaluated, and popped once written
        w = mx.load(str(source / shard))
        for k in sorted(w):
            v = w.pop(k)
            if k in writers or k == EMBED:
                f = np.array(v.astype(mx.float32))
                f = orthogonalize_embedding(f, r) if k == EMBED else orthogonalize_writer(f, r)
                e = mx.array(f).astype(v.dtype)
                mx.eval(e)
                del f
                edited += 1
                if k == EMBED:
                    put(LM_HEAD, v)
                put(k, e)
            else:
                mx.eval(v)
                put(k, v)
    flush()
    if edited != len(writers) + 1:
        raise RuntimeError(f"edited {edited} tensors, expected {len(writers) + 1}")
    index["weight_map"] = weight_map
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    cfg["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in source.iterdir():
        if f.suffix != ".safetensors" and not (out / f.name).exists():
            shutil.copy(f, out / f.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reuse-selection", action="store_true", help="skip fitting; use the saved selection"
    )
    ap.add_argument(
        "--select-only", action="store_true", help="print the layer table, write nothing"
    )
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / f"{KEY}-{TAG}"
    if (dest / "provenance.json").exists():
        raise SystemExit(f"{dest} exists")
    source = Path(snapshot_download(BASE_REPO, revision=BASE_SHA, local_files_only=True))
    sel_npy, sel_json = MODELS_DIR / f".{KEY}-selection.npy", MODELS_DIR / f".{KEY}-selection.json"
    if args.reuse_selection and sel_json.exists():
        r, record = np.load(sel_npy), json.loads(sel_json.read_text())
    else:
        r, record = fit_and_select(args)
        np.save(sel_npy, r)
        sel_json.write_text(json.dumps(record, indent=2))
    if args.select_only:
        print(json.dumps(record, indent=2))
        return
    bf16 = MODELS_DIR / f".{KEY}-bf16.partial"
    tmp = MODELS_DIR / f".{KEY}-{TAG}.partial"
    for d in (bf16, tmp):
        if d.exists():
            shutil.rmtree(d)
    write_ablated(source, r, bf16)
    # convert in a fresh process: the fitting step's freed memory is not returned to the OS
    # promptly and the conversion would thrash swap (same reason as harmonize_quant.smoke)
    subprocess.run(
        [
            "uv", "run", "mlx_lm.convert", "--hf-path", str(bf16), "--mlx-path", str(tmp), "-q",
            "--q-bits", str(TARGET["bits"]), "--q-group-size", str(TARGET["group_size"]),
        ],
        check=True,
    )  # fmt: skip
    shutil.rmtree(bf16)
    tmp.rename(dest)
    np.save(dest / "refusal_direction.npy", r)
    import mlx_lm

    prov = {
        "key": KEY,
        "source_repo": BASE_REPO,
        "source_sha": BASE_SHA,
        "license": "apache-2.0",
        "mlx_lm_version": mlx_lm.__version__,
        "quant": TARGET,
        "converted_on": date.today().isoformat(),
        "ablation": record,
        "size_gb": round(sum(f.stat().st_size for f in dest.glob("*.safetensors")) / 1e9, 2),
    }
    prov["tokenizer_edits"] = apply_chat_format(dest, "plain")
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"[{KEY}] smoke: {smoke(dest)!r}")


if __name__ == "__main__":
    main()
