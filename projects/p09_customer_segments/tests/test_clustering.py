import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from customer_segments import clustering
from customer_segments.evaluate import adjusted_eta2, select


def blobs():
    return make_blobs(n_samples=600, centers=4, cluster_std=0.6, random_state=0)


def test_kmeans_grid_and_stability_find_four_blobs():
    x, _ = blobs()
    grid, labels = clustering.kmeans_grid(x, [2, 3, 4, 5, 6], seed=0, n_init=5)
    assert grid.loc[grid["silhouette"].idxmax(), "k"] == 4
    assert len(np.unique(labels[4])) == 4
    stable = clustering.stability(x, 4, fits=5, seed=0, n_init=3)
    assert stable["stability_ari"] > 0.95 and stable["min_segment_jaccard"] > 0.9


def test_gap_statistic_prefers_structure():
    x, _ = blobs()
    gap = clustering.gap_statistic(x, [1, 2, 3, 4, 5, 6], references=5, seed=0)
    assert clustering.gap_choice(gap) == 4
    one_blob = np.random.default_rng(0).normal(size=(400, 2))
    assert clustering.gap_choice(clustering.gap_statistic(one_blob, [1, 2, 3], references=5, seed=0)) == 1


def test_density_methods_mark_noise():
    x, _ = blobs()
    x = np.vstack([x, np.random.default_rng(1).uniform(-15, 15, size=(30, 2))])
    hd, labels = clustering.hdbscan_grid(x, [25], 10, seed=0)
    assert hd["segments"].iat[0] == 4 and 0 < hd["noise_share"].iat[0] < 0.2
    db, _ = clustering.dbscan_grid(x, 10, [0.9], seed=0)
    assert db["segments"].iat[0] >= 1


def test_rfm_grid_covers_every_code():
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"recency_days": rng.uniform(0, 700, 1000), "frequency": rng.integers(1, 40, 1000)})
    segments = clustering.rfm_grid(table)
    assert segments.notna().all()
    assert {"champions", "hibernating", "new_customers", "at_risk"} <= set(segments)


def test_best_jaccard():
    a = np.array([0, 0, 1, 1])
    assert np.allclose(clustering.best_jaccard(a, np.array([5, 5, 7, 7])), [1, 1])
    assert np.allclose(clustering.best_jaccard(a, np.array([5, 5, 5, 7])), [2 / 3, 0.5])


def test_adjusted_eta2():
    y = np.array([0.0, 0.0, 1.0, 1.0] * 50)
    assert np.isclose(adjusted_eta2(y, y.astype(int)), 1.0)
    noise = np.random.default_rng(0).normal(size=2000)
    assert abs(adjusted_eta2(noise, np.arange(2000) % 5)) < 0.01
    assert adjusted_eta2(y, np.zeros(len(y))) == 0.0


def test_selection_rule():
    rows = []
    for scaling, stab in (("standard", 0.9), ("robust", 0.7)):
        for k, sil in ((3, 0.30), (4, 0.35), (5, 0.40)):
            rows.append({"scaling": scaling, "k": k, "silhouette": sil + (0.2 if scaling == "robust" else 0),
                         "stability_ari": stab if k < 5 else 0.6, "min_share": 0.05})  # fmt: skip
    table = pd.DataFrame(rows)
    table["candidate"] = (table["stability_ari"] >= 0.8) & (table["min_share"] >= 0.02)
    chosen = select(table)
    # Robust has higher silhouettes but lower stability; silhouettes are not compared across scalings.
    assert chosen["scaling"] == "standard" and chosen["k"] == 4
