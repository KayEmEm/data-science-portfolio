import numpy as np
from sklearn.metrics import average_precision_score

from flow_anomaly.metrics import (
    at_threshold,
    bootstrap,
    pr_auc_at_prevalence,
    precision_at_k,
    ranking,
    threshold_at_fpr,
)


def test_threshold_flags_the_target_share_of_normal_flows():
    normal = np.arange(1000, dtype=float)
    threshold = threshold_at_fpr(normal, 0.01)
    assert (normal > threshold).mean() <= 0.01
    assert (normal >= threshold).mean() >= 0.01


def test_at_threshold_counts():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    out = at_threshold(y, scores, 0.5)
    assert out["recall"] == 0.5
    assert np.isclose(out["fpr"], 1 / 3)
    assert out["precision"] == 0.5
    assert np.isnan(at_threshold(y, scores, 2.0)["precision"])


def test_precision_at_k_and_perfect_ranking():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert precision_at_k(y, scores, 2) == 1.0
    assert precision_at_k(y, scores, 10) == 0.5
    assert ranking(y, scores) == {"roc_auc": 1.0, "pr_auc": 1.0}


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    scores = y + rng.normal(0, 0.8, 2000)
    out = bootstrap(y, scores, 0.5, 50, seed=0)
    estimate = ranking(y, scores)["pr_auc"]
    assert out["pr_auc"][0] < estimate < out["pr_auc"][1]


def test_rescaled_pr_auc_matches_average_precision_at_the_observed_share():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.3).astype(int)
    scores = y * 0.8 + rng.normal(0, 1, 5000)
    observed = pr_auc_at_prevalence(y, scores, y.mean())
    assert np.isclose(observed, average_precision_score(y, scores), atol=1e-9)
    assert pr_auc_at_prevalence(y, scores, 0.01) < observed
