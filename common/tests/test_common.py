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
