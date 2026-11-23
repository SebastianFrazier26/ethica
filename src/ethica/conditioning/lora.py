"""LoRA conditioning via ``python -m mlx_lm lora`` (mlx-lm 0.32.2).

Rank/scale/dropout are not CLI flags in mlx-lm, only YAML keys (``lora_parameters``), so every
run is driven by a generated config file. A subprocess keeps training memory isolated from the
harness process and frees it fully on exit.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ethica.conditioning.corpus import (
    Chunk,
    Corpus,
    TokenCounter,
    approx_tokens,
    chunk,
    truncate_to_tokens,
)
from ethica.types import Conditioner, Message, ModelSpec

CHUNK_TOKENS = 512  # well under max_seq_length even if approx_tokens undercounts by ~2x
VALID_FRACTION = 0.05
MIN_VALID = 4


@dataclass(frozen=True)
class LoRAConfig:
    """Fixed across frameworks. Sized for a <=4B 4-bit model on 16 GB."""

    rank: int = 16
    scale: float = 20.0  # mlx-lm multiplies the LoRA delta by this (alpha/rank analogue)
    dropout: float = 0.0
    num_layers: int = 16
    batch_size: int = 2
    max_seq_length: int = 1024
    iters: int = 300
    learning_rate: float = 1e-4
    grad_checkpoint: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 100
    val_batches: int = 5
    save_every: int = 100
    seed: int = 0


def suggested_target_tokens(cfg: LoRAConfig, chunk_tokens: int = CHUNK_TOKENS) -> int:
    """Tokens consumed by exactly one pass over iters * batch_size samples."""
    return cfg.iters * cfg.batch_size * chunk_tokens


def build_lora_dataset(
    corpus: Corpus,
    out_dir: str | Path,
    target_tokens: int,
    seed: int = 0,
    chunk_tokens: int = CHUNK_TOKENS,
    count: TokenCounter = approx_tokens,
) -> dict[str, Any]:
    """Write ``train.jsonl`` / ``valid.jsonl`` ({"text": ...}) holding ``target_tokens`` of train.

    A seeded shuffle of the framework's chunks is cycled (upsampled) or cut (truncated) until the
    train split reaches target_tokens by ``count``; a small disjoint slice becomes valid. mlx-lm's
    training length is set by ``iters`` x ``batch_size``, so choose target_tokens near
    ``suggested_target_tokens`` for single-epoch exposure. Pass the model's tokenizer counter to
    match on real tokens rather than the chars/4 estimate.
    """
    chunks = chunk(corpus, chunk_tokens, 0, count)
    if not chunks:
        raise ValueError("corpus produced no chunks")
    rng = random.Random(seed)
    order: list[Chunk] = chunks[:]
    rng.shuffle(order)
    n_valid = max(1, int(len(order) * VALID_FRACTION)) if len(order) > 1 else 0
    train_pool = order[n_valid:] or order
    # mlx-lm raises if the valid set has fewer examples than batch_size; cycle if the corpus
    # is too small to hold out MIN_VALID distinct chunks.
    held = order[:n_valid] or order
    valid = [held[i % len(held)] for i in range(max(MIN_VALID, len(held)))]

    train: list[str] = []
    total = 0
    i = 0
    while total < target_tokens:
        c = train_pool[i % len(train_pool)]
        if i and i % len(train_pool) == 0:
            rng.shuffle(train_pool)
        text = truncate_to_tokens(c.text, target_tokens - total, count)
        if not text:
            break
        train.append(text)
        total += count(text)
        i += 1

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", [c.text for c in valid])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    meta = {
        "framework": corpus.manifest.name,
        "target_tokens": target_tokens,
        "train_tokens": total,
        "train_samples": len(train),
        "valid_samples": len(valid),
        "unique_train_chunks": len(train_pool),
        "epochs": round(total / max(1, sum(count(c.text) for c in train_pool)), 3),
        "seed": seed,
        "data_hash": data_hash(out),
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def data_hash(dataset_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("train.jsonl", "valid.jsonl"):
        h.update((dataset_dir / name).read_bytes())
    return h.hexdigest()


# mlx-lm 0.32 prints a rich table, not "Iter N: Train loss": rows are
# "<iter> <loss> <arrow> <tok/s> <tokens>" and "<iter> val <loss> <seconds>".
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TRAIN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+\S?\s*\d+\s+[\d.]+k?\s*$", re.MULTILINE)
_VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)", re.MULTILINE)
_PEAK = re.compile(r"^ETHICA_PEAK_GB ([\d.]+)", re.MULTILINE)

# The CLI does not report peak memory; wrapping main() lets us read MLX's own high-water mark
# (RSS under-reports Metal buffers on unified memory).
_RUNNER = (
    "import sys, mlx.core as mx\n"
    "from mlx_lm.lora import main\n"
    "sys.argv = ['mlx_lm.lora', '-c', sys.argv[1]]\n"
    "main()\n"
    "print('ETHICA_PEAK_GB', mx.get_peak_memory() / 1e9)\n"
)


def _write_config(
    path: Path,
    model: str,
    dataset_dir: Path,
    adapter_dir: Path,
    cfg: LoRAConfig,
    extra: Mapping[str, Any] | None = None,
) -> None:
    conf = {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(dataset_dir),
        "adapter_path": str(adapter_dir),
        "num_layers": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "iters": cfg.iters,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "grad_checkpoint": cfg.grad_checkpoint,
        "steps_per_report": cfg.steps_per_report,
        "steps_per_eval": cfg.steps_per_eval,
        "val_batches": cfg.val_batches,
        "save_every": cfg.save_every,
        "seed": cfg.seed,
        "lora_parameters": {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout},
    }
    clash = sorted(set(extra or {}) & set(conf))
    if clash:
        raise ValueError(f"extra mlx-lm keys override fixed ones: {clash}")
    conf.update(extra or {})
    path.write_text(yaml.safe_dump(conf, sort_keys=True), encoding="utf-8")


def train_lora(
    model_spec: ModelSpec | str,
    dataset_dir: str | Path,
    out_dir: str | Path,
    config: LoRAConfig | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Train an adapter; returns the adapter directory (holds adapters.safetensors).

    `extra` passes further mlx-lm settings (e.g. ``mask_prompt``) straight into its config. They
    are kept out of LoRAConfig because its fields are hashed into v0 adapter ids."""
    cfg = config or LoRAConfig()
    model = model_spec.hf_repo if isinstance(model_spec, ModelSpec) else model_spec
    ds = Path(dataset_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    conf_path = out / "lora_config.yaml"
    _write_config(conf_path, model, ds, out, cfg, extra)

    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(conf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.monotonic() - t0
    log = _ANSI.sub("", proc.stdout.replace("\r", "\n") + "\n" + proc.stderr)
    (out / "train.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0 or not (out / "adapters.safetensors").is_file():
        raise RuntimeError(f"mlx_lm lora failed (rc={proc.returncode}); tail:\n{log[-2000:]}")

    peaks = [float(x) for x in _PEAK.findall(log)]
    meta = {
        "model": model,
        "config": asdict(cfg),
        "extra": dict(extra or {}),
        "data_hash": data_hash(ds),
        "train_loss": [(int(i), float(v)) for i, v in _TRAIN.findall(log)],
        "val_loss": [(int(i), float(v)) for i, v in _VAL.findall(log)],
        "peak_mem_gb": max(peaks) if peaks else None,
        "wall_s": round(wall, 1),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


class LoRAConditioner:
    """Weights carry the conditioning; the backend loads ``adapter_path``.

    ``inner`` supports ``lora+prompt`` by layering a message-level conditioner on top.
    """

    def __init__(self, adapter_path: str | Path, inner: Conditioner | None = None) -> None:
        self.adapter_path = str(adapter_path)
        self.inner = inner

    def prepare(self, messages: list[Message]) -> list[Message]:
        return self.inner.prepare(messages) if self.inner else messages
