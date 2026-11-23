"""Name -> Benchmark registry. Instances are cheap; data is only touched in `load`."""

from __future__ import annotations

from ethica.benchmarks import capability, ethics, safety
from ethica.benchmarks.base import Benchmark, Source
from ethica.benchmarks.capability import GSM8KBenchmark, MMLUBenchmark, TruthfulQABenchmark
from ethica.benchmarks.dilemmas import OpenDilemmaBenchmark
from ethica.benchmarks.ethics import SUBSETS, EthicsBenchmark
from ethica.benchmarks.fidelity import FidelityBenchmark
from ethica.benchmarks.safety import HarmBenchBenchmark, XSTestBenchmark


def _build() -> dict[str, Benchmark]:
    benches: list[Benchmark] = [FidelityBenchmark(), OpenDilemmaBenchmark()]
    benches += [EthicsBenchmark(s) for s in SUBSETS]
    benches += [
        XSTestBenchmark("safe", default_n=100),
        XSTestBenchmark("unsafe", default_n=50),
        HarmBenchBenchmark(),
        GSM8KBenchmark(),
        MMLUBenchmark(),
        TruthfulQABenchmark(),
    ]
    return {b.name: b for b in benches}


REGISTRY: dict[str, Benchmark] = _build()


def get(name: str) -> Benchmark:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(REGISTRY)}") from None


def names() -> list[str]:
    return list(REGISTRY)


def all_sources() -> list[Source]:
    return [*ethics.SOURCES, *safety.SOURCES, *capability.SOURCES]
