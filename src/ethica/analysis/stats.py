"""Statistics implemented from scratch in numpy (DECISIONS D-14). All routines are seeded."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Exact sign-flip enumeration is used up to this many units (2**14 = 16384 sign patterns).
_EXACT_MAX_UNITS = 14
_TOL = 1e-12


class Interval(NamedTuple):
    mean: float
    lo: float
    hi: float
    n: int
    n_clusters: int


class PermResult(NamedTuple):
    diff: float  # mean(a) - mean(b) over pairs
    p: float
    n_pairs: int
    exact: bool


def _as_float(x: Sequence[float] | FloatArray) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def mean_ci_cluster_bootstrap(
    values: Sequence[float] | FloatArray,
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile CI for the mean, resampling whole clusters (e.g. tasks) with replacement.

    NaNs are dropped. Each replicate is the pooled mean (sum / count) over the resampled
    clusters, so unequal cluster sizes are weighted by their observation counts.
    """
    v = _as_float(values)
    if len(v) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    keep = ~np.isnan(v)
    v = v[keep]
    labels = [c for c, k in zip(clusters, keep, strict=True) if k]
    if len(v) == 0:
        return Interval(math.nan, math.nan, math.nan, 0, 0)
    uniq = {c: i for i, c in enumerate(dict.fromkeys(labels))}
    cid = np.fromiter((uniq[c] for c in labels), dtype=np.int64, count=len(labels))
    k = len(uniq)
    sums = np.bincount(cid, weights=v, minlength=k)
    counts = np.bincount(cid, minlength=k).astype(np.float64)
    point = float(v.mean())
    if k == 1:
        return Interval(point, point, point, len(v), 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), len(v), k)


def paired_permutation_test(
    a: Sequence[float] | FloatArray,
    b: Sequence[float] | FloatArray,
    pair_keys: Sequence[Hashable],
    n_perm: int = 10000,
    seed: int = 0,
    clusters: Sequence[Hashable] | None = None,
) -> PermResult:
    """Two-sided sign-flip test of mean(a - b) == 0 on paired observations.

    a[i] and b[i] are the two variants' outcomes on pair_keys[i] (task x seed). Rows with a
    NaN on either side are dropped; repeated keys are averaged. If `clusters` is given, signs
    are flipped per cluster (all pairs of a task flip together), which is the conservative
    choice when seeds of one task are correlated. Small designs are enumerated exactly;
    otherwise Monte-Carlo with the +1 correction p = (1 + #{|T*| >= |T|}) / (1 + n_perm).
    """
    av, bv = _as_float(a), _as_float(b)
    if not (len(av) == len(bv) == len(pair_keys)):
        raise ValueError("a, b and pair_keys must have equal length")
    if clusters is not None and len(clusters) != len(av):
        raise ValueError("clusters must match length of a")
    ok = ~(np.isnan(av) | np.isnan(bv))
    diffs: dict[Hashable, list[float]] = {}
    unit_of: dict[Hashable, Hashable] = {}
    for i in map(int, np.flatnonzero(ok)):
        key = pair_keys[i]
        diffs.setdefault(key, []).append(float(av[i] - bv[i]))
        unit_of[key] = clusters[i] if clusters is not None else key
    if not diffs:
        return PermResult(math.nan, math.nan, 0, False)
    pair_d = np.array([np.mean(v) for v in diffs.values()])
    n_pairs = len(pair_d)
    obs = float(pair_d.mean())
    if clusters is None:
        unit_sum = pair_d
    else:
        order = {u: j for j, u in enumerate(dict.fromkeys(unit_of.values()))}
        unit_sum = np.zeros(len(order))
        for key, d in zip(diffs, pair_d, strict=True):
            unit_sum[order[unit_of[key]]] += d
    u = len(unit_sum)
    thresh = abs(unit_sum.sum()) - _TOL * max(1.0, float(np.abs(unit_sum).sum()))

    if u <= _EXACT_MAX_UNITS:
        bits = (np.arange(2**u)[:, None] >> np.arange(u)) & 1
        stats = np.abs((1 - 2 * bits) @ unit_sum)
        return PermResult(obs, float((stats >= thresh).mean()), n_pairs, True)

    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = n_perm
    while remaining > 0:
        m = min(remaining, 2000)
        signs = rng.integers(0, 2, size=(m, u)) * 2 - 1
        exceed += int((np.abs(signs @ unit_sum) >= thresh).sum())
        remaining -= m
    return PermResult(obs, (1 + exceed) / (1 + n_perm), n_pairs, False)


def holm_bonferroni(pvals: Sequence[float] | FloatArray) -> FloatArray:
    """Holm step-down adjusted p-values (FWER). NaNs pass through and do not count toward m."""
    p = _as_float(pvals)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(pvals: Sequence[float] | FloatArray) -> FloatArray:
    """Benjamini-Hochberg adjusted p-values (FDR). NaNs pass through."""
    p = _as_float(pvals)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    scaled = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(scaled[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def risk_difference(p1: float, p2: float) -> float:
    return p1 - p2


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions: 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def wilson_ci(k: float, n: float, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # exact at the boundaries: centre - half is 0 only up to rounding, which varies by libm
    lo = 0.0 if k <= 0 else max(0.0, centre - half)
    hi = 1.0 if k >= n else min(1.0, centre + half)
    return (lo, hi)
