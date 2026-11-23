import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".npz", ".bin")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def test_no_weights_tracked():
    files = _git("ls-files").stdout.splitlines()
    assert files, "not a git checkout"
    assert [f for f in files if f.endswith(WEIGHT_SUFFIXES)] == []


@pytest.mark.parametrize(
    "path", ["adapters/v1/x/adapters.safetensors", "models/x", "runs/x", "labels/x", "data/v1/x"]
)
def test_local_artifacts_ignored(path):
    # v1 trains models without safeguards; nothing they produce may be committed (proposal §6)
    assert _git("check-ignore", "-q", path).returncode == 0, path
