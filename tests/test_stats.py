import math

import numpy as np
import pytest

from ethica.analysis.stats import (
    benjamini_hochberg,
    cohens_h,
    holm_bonferroni,
    mean_ci_cluster_bootstrap,
    paired_permutation_test,
    risk_difference,
    wilson_ci,
)


def test_wilson_known_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)
    assert math.isnan(wilson_ci(0, 0)[0])


def test_effect_sizes():
    assert risk_difference(0.5, 0.2) == pytest.approx(0.3)
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(1.0, 0.0) == pytest.approx(math.pi)
    assert cohens_h(0.7, 0.4) == pytest.approx(-cohens_h(0.4, 0.7))


def test_holm_textbook():
    adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_bh_textbook():
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_adjust_nan_and_cap():
    adj = holm_bonferroni([0.5, math.nan, 0.9])
    assert adj[0] == pytest.approx(1.0)
    assert math.isnan(adj[1])
    assert adj[2] == pytest.approx(1.0)


def test_permutation_identical_is_one():
    x = np.random.default_rng(0).random(40)
    r = paired_permutation_test(x, x.copy(), list(range(40)), seed=1)
    assert r.p == pytest.approx(1.0)
    assert r.diff == 0.0


def test_permutation_shifted_small_p_and_deterministic():
    rng = np.random.default_rng(0)
    b = rng.random(40)
    a = b + 0.5 + rng.normal(0, 0.05, 40)
    keys = list(range(40))
    r1 = paired_permutation_test(a, b, keys, n_perm=2000, seed=3)
    r2 = paired_permutation_test(a, b, keys, n_perm=2000, seed=3)
    assert r1 == r2
    assert r1.p < 0.002
    assert not r1.exact


def test_permutation_exact_small():
    # 3 pairs all +1: only the two all-same-sign patterns reach |T| = 3 => p = 2/8
    r = paired_permutation_test([1, 1, 1], [0, 0, 0], ["a", "b", "c"])
    assert r.exact
    assert r.p == pytest.approx(0.25)


def test_permutation_drops_nan_and_cluster_flip():
    a = [1.0, 1.0, math.nan, 1.0] * 5
    b = [0.0, 0.0, 0.0, 0.0] * 5
    keys = list(range(20))
    r = paired_permutation_test(a, b, keys, seed=0)
    assert r.n_pairs == 15
    clusters = [i // 4 for i in range(20)]
    rc = paired_permutation_test(a, b, keys, seed=0, clusters=clusters)
    assert rc.p >= r.p  # task-level flipping is never less conservative here


def test_bootstrap_basic_and_nan():
    vals = [0, 1, 1, 0, math.nan, 1]
    cl = ["a", "a", "b", "b", "c", "c"]
    iv = mean_ci_cluster_bootstrap(vals, cl, n_boot=500, seed=0)
    assert iv.mean == pytest.approx(0.6)
    assert iv.n == 5
    assert iv.lo <= iv.mean <= iv.hi
    assert mean_ci_cluster_bootstrap(vals, cl, n_boot=500, seed=0) == iv
    assert math.isnan(mean_ci_cluster_bootstrap([math.nan], ["a"]).mean)


def test_bootstrap_coverage_with_cluster_effects():
    rng = np.random.default_rng(42)
    n_clusters, per = 25, 4
    covered, trials = 0, 150
    cl = np.repeat(np.arange(n_clusters), per)
    for t in range(trials):
        p_c = rng.beta(2, 2, n_clusters)  # true mean 0.5
        vals = (rng.random(n_clusters * per) < p_c[cl]).astype(float)
        iv = mean_ci_cluster_bootstrap(vals, cl.tolist(), n_boot=300, seed=t)
        covered += iv.lo <= 0.5 <= iv.hi
    assert 0.88 <= covered / trials <= 1.0
