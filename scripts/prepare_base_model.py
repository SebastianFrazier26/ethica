"""Convert the v1 base checkpoint to affine 4-bit g64 and set its chat format (D-61, D-65).

    uv run python scripts/prepare_base_model.py                      # ChatML markers
    uv run python scripts/prepare_base_model.py --chat-format plain  # ordinary-token fallback
    uv run python scripts/prepare_base_model.py --chat-format plain --reformat   # no reconversion

Output: models/qwen3-4b-base-q4g64/ + provenance.json (models/ is gitignored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date

from harmonize_quant import MODELS_DIR, TAG, TARGET, smoke, wait_for_ram

from ethica.training.chat_format import apply_chat_format

KEY = "qwen3-4b-base"
REPO = "Qwen/Qwen3-4B-Base"
SHA = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
SOURCE_GB = 8.04


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-format", choices=["chatml", "plain"], default="chatml")
    ap.add_argument("--reformat", action="store_true", help="only rewrite the chat format")
    args = ap.parse_args()
    dest = MODELS_DIR / f"{KEY}-{TAG}"
    prov_path = dest / "provenance.json"
    if args.reformat:
        prov = json.loads(prov_path.read_text())
    else:
        if prov_path.exists():
            raise SystemExit(f"{dest} exists; use --reformat to change only the chat format")
        import mlx_lm
        from huggingface_hub import snapshot_download
        from mlx_lm import convert

        wait_for_ram(SOURCE_GB * 0.4 + 1.5)
        MODELS_DIR.mkdir(exist_ok=True)
        tmp = MODELS_DIR / f".{KEY}-{TAG}.partial"
        if tmp.exists():
            subprocess.run(["rm", "-rf", str(tmp)], check=True)
        # a local path, not the repo id: convert's own download asks the Hub for file patterns
        # the checkpoint does not have and fails when offline
        source = snapshot_download(REPO, revision=SHA, local_files_only=True)
        convert(
            source,
            mlx_path=str(tmp),
            quantize=True,
            q_bits=TARGET["bits"],
            q_group_size=TARGET["group_size"],
            q_mode=TARGET["mode"],
        )
        tmp.rename(dest)
        prov = {
            "key": KEY,
            "source_repo": REPO,
            "source_sha": SHA,
            "license": "apache-2.0",
            "mlx_lm_version": mlx_lm.__version__,
            "quant": TARGET,
            "converted_on": date.today().isoformat(),
            "size_gb": round(sum(f.stat().st_size for f in dest.glob("*.safetensors")) / 1e9, 2),
        }
    prov["tokenizer_edits"] = apply_chat_format(dest, args.chat_format)
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"[{KEY}] {prov['tokenizer_edits']} at {dest}")
    if not args.reformat:
        print(f"[{KEY}] smoke: {smoke(dest)!r}")


if __name__ == "__main__":
    main()
