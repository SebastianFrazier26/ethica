"""Chat-format training files for one arm, from screened teacher answers.

Every arm is built from the same prompt ids in the same order with the same split, so two arms'
datasets differ only in the assistant turns.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ethica.conditioning.corpus import approx_tokens
from ethica.conditioning.lora import data_hash
from ethica.training.generate import read_rows

MIN_VALID = 4  # mlx-lm needs at least batch_size validation rows


def common_prompt_ids(arm_files: Sequence[Path]) -> list[str]:
    """Prompt ids kept (not rejected) in every arm, in the first file's order."""
    if not arm_files:
        raise ValueError("no arm files")
    kept = [[r["prompt_id"] for r in read_rows(f) if r["reject"] is None] for f in arm_files]
    shared = set(kept[0]).intersection(*kept[1:])
    return [pid for pid in kept[0] if pid in shared]


def build_chat_dataset(
    arm_file: Path,
    out_dir: Path,
    keep_ids: Sequence[str],
    seed: int = 0,
    valid_fraction: float = 0.05,
    shared_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """`shared_rows` ({"id", "messages"}, e.g. toolformat.tool_rows) are added unchanged to
    every arm, split by the same rule as the arm's own rows."""
    rows = {r["prompt_id"]: r for r in read_rows(arm_file) if r["reject"] is None}
    missing = [pid for pid in keep_ids if pid not in rows]
    if missing:
        raise ValueError(
            f"{arm_file.name}: {len(missing)} requested prompts have no kept answer "
            f"(first: {missing[0]})"
        )
    if len(keep_ids) < 2 * MIN_VALID:
        raise ValueError(f"need at least {2 * MIN_VALID} examples, got {len(keep_ids)}")
    # the split depends on ids and seed only, never on the answers, so it is the same in every arm
    shared = {str(r["id"]): r["messages"] for r in shared_rows}
    if len(shared) != len(shared_rows) or set(shared) & set(keep_ids):
        raise ValueError("shared row ids must be unique and distinct from prompt ids")
    order = sorted([*keep_ids, *shared])
    random.Random(seed).shuffle(order)
    n_valid = max(MIN_VALID, round(len(order) * valid_fraction))
    splits = {"valid": order[:n_valid], "train": order[n_valid:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = 0
    for name, ids in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for rid in ids:
                msgs = shared.get(rid) or [
                    {"role": "user", "content": rows[rid]["prompt"]},
                    {"role": "assistant", "content": rows[rid]["answer"]},
                ]
                tokens += sum(approx_tokens(m["content"]) for m in msgs)
                fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    arms = {r["arm"] for r in rows.values()}
    meta = {
        "arm": sorted(arms)[0] if len(arms) == 1 else sorted(arms),
        "n_train": len(splits["train"]),
        "n_valid": n_valid,
        "n_shared": len(shared),
        "approx_tokens": tokens,
        "seed": seed,
        "data_hash": data_hash(out_dir),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
