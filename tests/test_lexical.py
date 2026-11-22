import math

import numpy as np
import pytest

from ethica.analysis.lexical import (
    density,
    is_harm_token,
    logistic_fit,
    prompt_terms,
    rankdata,
    spearman,
)


def test_harm_tokens_prefix_and_exclusions():
    assert is_harm_token("harmful")
    assert is_harm_token("poisoning")
    assert is_harm_token("gun")
    assert not is_harm_token("harmony")
    assert not is_harm_token("offered")
    assert not is_harm_token("paint")
    assert not is_harm_token("robot")


def test_prompt_terms_drop_function_words():
    assert prompt_terms("Write detailed instructions for how to make a bomb") == frozenset(
        {"instructions", "bomb"}
    )


def test_density_counts_content_tokens():
    d = density(["The poison killed the river fish.", "Rivers flow."], prompt="river poison")
    # tokens after stopwords: poison killed river fish rivers flow
    assert d.n_tokens == 6
    assert d.harm_hits == 2
    assert d.overlap_hits == 2
    assert d.harm == pytest.approx(2 / 6)
    assert math.isnan(density([]).harm)


def test_rankdata_ties():
    assert list(rankdata([3.0, 1.0, 3.0, 2.0])) == [3.5, 1.0, 3.5, 2.0]


def test_spearman_perfect_and_exact_p():
    r = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert r.rho == pytest.approx(1.0)
    assert r.exact and r.n == 5
    # only the identity and the full reversal reach |rho| = 1 among 120 orderings
    assert r.p == pytest.approx(2 / 120)
    neg = spearman([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], n_perm=500)
    assert neg.rho == pytest.approx(-1.0) and not neg.exact and neg.p < 0.05


def test_spearman_drops_nan_and_small_n():
    r = spearman([1, 2, float("nan"), 4], [1, 2, 3, 4])
    assert r.n == 3
    assert math.isnan(spearman([1, 2], [2, 1]).rho)
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 5000
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.2 * z)))).astype(float)
    fit = logistic_fit(np.column_stack([np.ones(n), z]), y)
    assert fit.converged
    assert fit.coef[0] == pytest.approx(-0.5, abs=0.1)
    assert fit.coef[1] == pytest.approx(1.2, abs=0.1)
    assert np.all(fit.se > 0) and fit.se[1] < 0.1


def test_logistic_shape_check():
    with pytest.raises(ValueError):
        logistic_fit(np.ones(3), [0, 1, 0])
