"""Download every pinned dataset file into data/cache/ (idempotent)."""

from __future__ import annotations

from pathlib import Path

from ethica.benchmarks.base import fetch
from ethica.benchmarks.registry import all_sources


def prefetch_all() -> list[Path]:
    return [fetch(s) for s in all_sources()]
