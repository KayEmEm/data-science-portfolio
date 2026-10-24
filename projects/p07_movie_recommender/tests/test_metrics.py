import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from movie_recs.metrics import PAD, novelty, per_user, summarize, top_n


def test_recall_and_ndcg_by_hand():
    users = np.array([1, 2])
    recs = np.array([[10, 11, 12], [20, 21, PAD]])
    targets = pd.DataFrame({"user": [1, 1, 2], "item": [12, 13, 99]})
    scores = per_user(users, recs, targets, k=3)
    assert scores["recall"].tolist() == [0.5, 0.0]
    # one hit at rank 3 out of two liked movies: (1/log2 4) / (1 + 1/log2 3)
    expected = (1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert scores["ndcg"].iloc[0] == pytest.approx(expected)
    assert scores["ndcg"].iloc[1] == 0.0


def test_summary_coverage_and_hit_rate():
    users = np.array([1, 2])
    recs = np.array([[10, 11], [10, 12]])
    targets = pd.DataFrame({"user": [1, 2], "item": [11, 99]})
    popularity = pd.Series({10: 50, 11: 5, 12: 1})
    result = summarize(users, recs, targets, popularity, n_history_users=100, n_catalog=4, k=2)
    assert result["hit_rate@2"] == 0.5
    assert result["coverage@2"] == 0.75
    assert result["recall@2"] == 0.5


def test_novelty_is_higher_for_rare_items():
    popularity = pd.Series({1: 100, 2: 1})
    assert novelty(np.array([[2]]), popularity, 100) > novelty(np.array([[1]]), popularity, 100)
    assert novelty(np.array([[1]]), popularity, 100) == 0.0


def test_top_n_skips_seen_items_and_pads():
    items = np.array([100, 101, 102, 103])
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=np.float32))
    scores = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    found, values = top_n(lambda rows: scores[rows], np.array([0, 1]), seen, items, n=2)
    assert found[0].tolist() == [101, 102]
    assert found[1].tolist() == [103, PAD]
    assert np.isneginf(values[1, 1])
