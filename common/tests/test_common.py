import numpy as np

from portfolio_common.evaluation import binary_metrics
from portfolio_common.seed import set_seed


def test_set_seed_is_reproducible():
    a = set_seed(7).random(3)
    b = set_seed(7).random(3)
    np.testing.assert_array_equal(a, b)


def test_binary_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert 0.0 <= m["brier"] < 0.05


def test_sample_keeps_whole_units_and_is_nested():
    import pandas as pd

    from portfolio_common.sample import select

    keys = pd.Series([u for u in range(200) for _ in range(3)])
    strata = keys % 2
    small = select(keys, 0.1, seed=1, strata=strata)
    large = select(keys, 0.3, seed=1, strata=strata)
    assert (small <= large).all()
    assert pd.Series(small).groupby(keys).nunique().max() == 1  # a unit is kept whole or not at all
    assert keys[small].nunique() == 20 and keys[large].nunique() == 60
    assert (select(keys, 0.1, seed=1, strata=strata) == small).all()
    assert not (select(keys, 0.1, seed=2, strata=strata) == small).all()
    assert select(keys, 1.0).all()
    assert select(keys, 0.01, always=[7])[keys == 7].all()
