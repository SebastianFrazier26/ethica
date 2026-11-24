"""Study v1 data and training pipeline (docs/study-v1/PROPOSAL.md).

    uv run python scripts/v1_pipeline.py prompts
    uv run python scripts/v1_pipeline.py generate generic [--limit 50]
    uv run python scripts/v1_pipeline.py revise virtue      # optional critique and revise
    uv run python scripts/v1_pipeline.py dataset generic [placebo virtue virtue.revised ...]
    uv run python scripts/v1_pipeline.py train generic --seed 0

`generate`, `revise` and `train` load a model: run them only when no other MLX job is on the
machine.
Everything is written under data/v1/ and adapters/v1/, both gitignored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ethica.backends.registry import get_model, make_backend
from ethica.conditioning.corpus import framework_from_manifest, load_corpus
from ethica.conditioning.lora import LoRAConfig, train_lora
from ethica.conditioning.rag import RAGConditioner, build_index
from ethica.training.dataset import build_chat_dataset, common_prompt_ids
from ethica.training.generate import ArmSpec, generate_arm
from ethica.training.prompts import TrainPrompt, build_prompt_set
from ethica.training.revise import revise_arm
from ethica.training.toolformat import tool_rows
from ethica.types import Conditioner, GenerationParams

ROOT = Path(__file__).resolve().parent.parent
WRAPPERS = {"caffeinate", "nohup", "sh", "zsh", "bash", "uv"}
MLX_JOB_PATTERNS = (
    "mlx_lm",
    "ethica run",
    "v1_pipeline.py generate",
    "v1_pipeline.py revise",
    "v1_pipeline.py train",
)


def load_plan(path: Path) -> dict[str, Any]:
    plan: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return plan


def arm_spec(plan: dict[str, Any], name: str) -> ArmSpec:
    row = next((a for a in plan["arms"] if a["name"] == name), None)
    if row is None:
        raise SystemExit(f"unknown arm {name!r}; known: {[a['name'] for a in plan['arms']]}")
    commitments = None
    if row.get("commitments"):
        path = ROOT / row["commitments"]
        if not path.is_file():
            raise SystemExit(
                f"arm {name!r}: {row['commitments']} is missing. The framework commitments "
                "need the owner's sign-off before this arm is generated (D-64)."
            )
        commitments = path.read_text(encoding="utf-8")
    return ArmSpec(name, row["kind"], row.get("corpus"), commitments)


def other_mlx_jobs() -> list[str]:
    """Other processes on the GPU. This process and its ancestors (uv, shells) are excluded:
    their command lines contain this script's own name."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True
    ).stdout
    procs = {}
    for line in out.splitlines():
        pid, ppid, cmd = line.split(None, 2)
        procs[int(pid)] = (int(ppid), cmd)
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:
        mine.add(pid)
        pid = procs[pid][0]
    return [
        f"{pid} {cmd[:110]}"
        for pid, (_, cmd) in procs.items()
        if pid not in mine
        # wrappers quote the job's command line but hold no model; the job itself is their child
        and Path(cmd.split()[0]).name not in WRAPPERS
        and any(p in cmd for p in MLX_JOB_PATTERNS)
    ]


def prompts(plan: dict[str, Any]) -> list[TrainPrompt]:
    ps = build_prompt_set(plan["n_prompts"], plan["prompt_seed"])
    path = ROOT / plan["data_root"] / "prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(p.__dict__, ensure_ascii=False) + "\n" for p in ps), encoding="utf-8"
    )
    return ps


def cmd_generate(plan: dict[str, Any], name: str, limit: int | None) -> None:
    arm = arm_spec(plan, name)
    ps = prompts(plan)[:limit]
    conditioner: Conditioner | None = None
    if arm.corpus:
        corpus = load_corpus(ROOT / "corpora" / arm.corpus)
        fw = framework_from_manifest(corpus.manifest, corpus.root)
        conditioner = RAGConditioner(fw, build_index(corpus), plan["rag_k"], plan["budget_tokens"])
    backend = make_backend(get_model(plan["teacher"]))
    try:
        stats = generate_arm(
            backend,
            arm,
            ps,
            ROOT / plan["data_root"] / "answers" / f"{name}.jsonl",
            GenerationParams(**plan["generation"]),
            conditioner,
        )
    finally:
        backend.close()
    print(f"{name}: {stats}")


def cmd_revise(plan: dict[str, Any], name: str) -> None:
    arm = arm_spec(plan, name)
    answers = ROOT / plan["data_root"] / "answers"
    backend = make_backend(get_model(plan["teacher"]))
    try:
        stats = revise_arm(
            backend,
            arm,
            answers / f"{name}.jsonl",
            answers / f"{name}.revised.jsonl",
            GenerationParams(**plan["generation"]),
        )
    finally:
        backend.close()
    print(f"{name}.revised: {stats}")


def cmd_dataset(plan: dict[str, Any], names: list[str]) -> None:
    root = ROOT / plan["data_root"]
    files = [root / "answers" / f"{n}.jsonl" for n in names]
    ids = common_prompt_ids(files)
    shared = tool_rows(plan.get("tool_tasks", 0), plan.get("tool_seed", 0))
    for n, f in zip(names, files, strict=True):
        print(n, build_chat_dataset(f, root / "datasets" / n, ids, shared_rows=shared))


def cmd_train(plan: dict[str, Any], name: str, seed: int) -> None:
    ds = ROOT / plan["data_root"] / "datasets" / name
    meta = json.loads((ds / "dataset_meta.json").read_text())
    lora = dict(plan["lora"])
    iters = math.ceil(plan["epochs"] * meta["n_train"] / lora["batch_size"])
    cfg = LoRAConfig(**lora, iters=iters, seed=seed)
    out = train_lora(
        get_model(plan["base_model"]),
        ds,
        ROOT / plan["adapters_root"] / f"{name}-s{seed}",
        cfg,
        extra=plan.get("mlx_extra"),
    )
    tm = json.loads((out / "train_meta.json").read_text())
    print(f"{name}-s{seed}: {iters} iters, {tm['wall_s']} s, peak {tm['peak_mem_gb']} GB")
    print("val loss:", tm["val_loss"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/training/v1-pilot.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    g = sub.add_parser("generate")
    g.add_argument("arm")
    g.add_argument("--limit", type=int)
    r = sub.add_parser("revise")
    r.add_argument("arm")
    d = sub.add_parser("dataset")
    d.add_argument("arms", nargs="+")
    t = sub.add_parser("train")
    t.add_argument("arm")
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    plan = load_plan(args.plan)
    if args.cmd in ("generate", "revise", "train") and (jobs := other_mlx_jobs()):
        raise SystemExit("another MLX job is running; not starting:\n  " + "\n  ".join(jobs))
    if args.cmd == "prompts":
        print(f"wrote {len(prompts(plan))} prompts")
    elif args.cmd == "generate":
        cmd_generate(plan, args.arm, args.limit)
    elif args.cmd == "revise":
        cmd_revise(plan, args.arm)
    elif args.cmd == "dataset":
        cmd_dataset(plan, args.arms)
    else:
        cmd_train(plan, args.arm, args.seed)


if __name__ == "__main__":
    main()
