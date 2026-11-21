"""Cache all static-benchmark datasets under data/cache/.

Usage: uv run python scripts/fetch_benchmarks.py
"""

from ethica.benchmarks.prefetch import prefetch_all

if __name__ == "__main__":
    for p in prefetch_all():
        print(p)
