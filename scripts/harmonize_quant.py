"""Re-quantise stock/abliterated pair members to one identical scheme (affine 4-bit, group 64).

Why: quantisation differences confound the safeguards contrast (docs/DECISIONS.md D-33).
Output: models/<key>-q4g64/ + provenance.json; wire up via `local_path` in configs/models.yaml.

    uv run python scripts/harmonize_quant.py llama-3.2-3b-abl qwen3-1.7b
    uv run python scripts/harmonize_quant.py --check      # print quant block of every pair member
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TAG = "q4g64"
TARGET: dict[str, Any] = {"bits": 4, "group_size": 64, "mode": "affine"}

# Full-precision source per key (first ungated entry wins). Stock keys use the upstream repo;
# abliterated keys use the abliteration author's bf16 repo.
SOURCES: dict[str, list[str]] = {
    "qwen3-1.7b": ["Qwen/Qwen3-1.7B"],
    "qwen3-1.7b-abl": ["huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2"],
    "llama-3.2-3b": ["meta-llama/Llama-3.2-3B-Instruct"],  # gated -> keep mlx-community q4
    "llama-3.2-3b-abl": ["huihui-ai/Llama-3.2-3B-Instruct-abliterated"],
    "phi-4-mini": ["microsoft/Phi-4-mini-instruct"],
    "phi-4-mini-abl": ["lunahr/Phi-4-mini-instruct-abliterated"],
    "granite-4.1-3b": ["ibm-granite/granite-4.1-3b"],
    "granite-4.1-3b-abl": ["huihui-ai/Huihui-granite-4.1-3b-abliterated"],
    "smollm3-3b": ["HuggingFaceTB/SmolLM3-3B"],
    "smollm3-3b-abl": ["mlx-community/SmolLM3-3B-abliterated-bf16"],
}


def _norm_quant(q: dict[str, Any] | None) -> dict[str, Any] | None:
    if not q:
        return None
    # older mlx-community configs omit "mode"; affine was the only mode then
    return {
        "bits": q.get("bits"),
        "group_size": q.get("group_size"),
        "mode": q.get("mode", "affine"),
    }


def available_gb() -> float:
    """Reclaimable RAM in GB: free + inactive + speculative + purgeable pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    pages = 0
    for line in out.splitlines():
        if line.split(":")[0] in {
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        }:
            pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page / 1e9


def wait_for_ram(need_gb: float, timeout_s: int = 3600) -> None:
    deadline = time.time() + timeout_s
    while (avail := available_gb()) < need_gb:
        if time.time() > deadline:
            raise TimeoutError(f"only {avail:.1f} GB reclaimable, need {need_gb:.1f}")
        print(f"  RAM tight ({avail:.1f} GB < {need_gb:.1f}); waiting 60s", flush=True)
        time.sleep(60)


def pick_source(key: str) -> tuple[str, str, float] | tuple[None, str, float]:
    """Return (repo, sha, size_gb) of the first reachable ungated source, else (None, reason, 0)."""
    api = HfApi()
    reasons = []
    for repo in SOURCES[key]:
        try:
            info = api.model_info(repo, files_metadata=True)
        except GatedRepoError:
            reasons.append(f"{repo}: gated")
            continue
        except RepositoryNotFoundError:
            reasons.append(f"{repo}: not found")
            continue
        if info.gated:
            reasons.append(f"{repo}: gated")
            continue
        size = sum(
            (s.size or 0) for s in info.siblings or [] if s.rfilename.endswith(".safetensors")
        )
        assert info.sha is not None
        return repo, info.sha, size / 1e9
    return None, "; ".join(reasons), 0.0


_SMOKE = (
    "import sys; from mlx_lm import generate, load; m, t, *_ = load(sys.argv[1]); "
    "print(generate(m, t, prompt='The capital of France is', max_tokens=20))"
)


def smoke(path: Path) -> str:
    # fresh process: in-process after convert, the freed-but-unreturned conversion memory made the
    # load thrash swap on a 16 GB machine
    out = subprocess.run(
        [sys.executable, "-c", _SMOKE, str(path)], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().replace("\n", " ")


def harmonize(key: str, stock_repo: str | None, do_smoke: bool) -> int:
    import mlx_lm
    from mlx_lm import convert

    dest = MODELS_DIR / f"{key}-{TAG}"
    if (dest / "provenance.json").exists():
        print(f"[{key}] skip: provenance exists")
        return 0
    repo, sha, size_gb = pick_source(key)
    if repo is None:
        print(f"[{key}] SKIP: no ungated full-precision source ({sha})")
        if stock_repo:
            cfg = json.loads(Path(hf_hub_download(stock_repo, "config.json")).read_text())
            q = _norm_quant(cfg.get("quantization"))
            verdict = "MATCHES target; keep it" if q == TARGET else f"DOES NOT match target ({q})"
            print(f"[{key}] fallback {stock_repo} quant {q}: {verdict}")
        return 2
    wait_for_ram(size_gb * 0.4 + 1.5)
    MODELS_DIR.mkdir(exist_ok=True)
    tmp = MODELS_DIR / f".{key}-{TAG}.partial"
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    print(f"[{key}] converting {repo}@{sha[:8]} ({size_gb:.1f} GB source)", flush=True)
    convert(
        repo,
        mlx_path=str(tmp),
        quantize=True,
        q_bits=TARGET["bits"],
        q_group_size=TARGET["group_size"],
        q_mode=TARGET["mode"],
        revision=sha,
    )
    got = _norm_quant(json.loads((tmp / "config.json").read_text()).get("quantization"))
    if got != TARGET:
        raise RuntimeError(f"{key}: produced quant {got}, wanted {TARGET}")
    tmp.rename(dest)
    out_gb = sum(f.stat().st_size for f in dest.glob("*.safetensors")) / 1e9
    prov = {
        "key": key,
        "source_repo": repo,
        "source_sha": sha,
        "mlx_lm_version": mlx_lm.__version__,
        "quant": TARGET,
        "dtype": "from source config (bfloat16)",
        "converted_on": date.today().isoformat(),
        "size_gb": round(out_gb, 2),
    }
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"[{key}] done: {out_gb:.2f} GB at {dest}")
    if do_smoke:
        print(f"[{key}] smoke: {smoke(dest)!r}")
    return 0


def check() -> None:
    import yaml

    rows = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    for r in rows:
        if r["key"] in ("tiny", "judge-default"):
            continue
        if r.get("local_path"):
            cfg_path = ROOT / r["local_path"] / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            where = r["local_path"]
        else:
            cfg = json.loads(Path(hf_hub_download(r["hf_repo"], "config.json")).read_text())
            where = r["hf_repo"]
        print(f"{r['key']:20s} {_norm_quant(cfg.get('quantization'))}  <- {where}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return 0
    import yaml

    repos = {
        r["key"]: r["hf_repo"]
        for r in yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    }
    rc = 0
    for key in args.keys:  # sequential on purpose: one conversion resident at a time
        if key not in SOURCES:
            print(f"[{key}] unknown key", file=sys.stderr)
            rc = 1
            continue
        rc = max(rc, harmonize(key, repos.get(key), not args.no_smoke))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
