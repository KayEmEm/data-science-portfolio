"""Clustering methods, quality measures and the rule based RFM baseline.

* k means over a range of k with inertia, silhouette, Calinski Harabasz and Davies Bouldin scores;
* the gap statistic (Tibshirani, Walther and Hastie, 2001) with uniform reference data in the
  principal component box of the data;
* bootstrap stability: k means refitted on resampled customers, every customer assigned by the
  refit, and the result compared with the full fit by adjusted Rand index and, per segment, by the
  Jaccard overlap of its best match (Hennig, 2007);
* DBSCAN and HDBSCAN, whose noise points form a group of their own;
* the RFM grid: recency and frequency scores from 1 to 5 mapped to ten named segments, the rule set
  that marketing teams use without any model.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

NOISE = -1
SILHOUETTE_ROWS = 6000

# Recency score then frequency score, first match wins.
RFM_GRID = [
    (r"5[4-5]", "champions"),
    (r"[3-4][4-5]", "loyal"),
    (r"[4-5][2-3]", "potential_loyalists"),
    (r"51", "new_customers"),
    (r"41", "promising"),
    (r"33", "need_attention"),
    (r"3[1-2]", "about_to_sleep"),
    (r"[1-2]5", "cannot_lose"),
    (r"[1-2][3-4]", "at_risk"),
    (r"[1-2][1-2]", "hibernating"),
]


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)


def silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    keep = labels != NOISE
    if len(np.unique(labels[keep])) < 2:
        return float("nan")
    size = min(int(keep.sum()), SILHOUETTE_ROWS)
    return float(silhouette_score(x[keep], labels[keep], sample_size=size, random_state=seed))


def quality(x: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    keep = labels != NOISE
    groups = np.unique(labels[keep])
    shares = pd.Series(labels).value_counts(normalize=True)
    out = {
        "segments": len(groups),
        "noise_share": float((~keep).mean()),
        "min_share": float(shares.drop(NOISE, errors="ignore").min()) if len(groups) else float("nan"),
        "silhouette": silhouette(x, labels, seed),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
    }
    if len(groups) >= 2:
        out["calinski_harabasz"] = float(calinski_harabasz_score(x[keep], labels[keep]))
        out["davies_bouldin"] = float(davies_bouldin_score(x[keep], labels[keep]))
    return out


def kmeans_grid(x: np.ndarray, ks: list[int], seed: int, n_init: int) -> tuple[pd.DataFrame, dict]:
    rows, labels = [], {}
    for k in ks:
        model = kmeans(x, k, seed, n_init)
        labels[k] = model.labels_
        rows.append({"k": k, "inertia": float(model.inertia_), **quality(x, model.labels_, seed)})
    return pd.DataFrame(rows), labels


def gap_statistic(x: np.ndarray, ks: list[int], references: int, seed: int) -> pd.DataFrame:
    """Gap(k) = mean log W of reference data minus log W of the data, with its standard error s(k)."""
    rng = np.random.default_rng(seed)
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vt.T
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    rows = []
    for k in ks:
        log_w = np.log(kmeans(x, k, seed, 3).inertia_)
        ref = []
        for _ in range(references):
            sample = rng.uniform(low, high, size=rotated.shape) @ vt
            ref.append(np.log(kmeans(sample, k, seed, 3).inertia_))
        ref = np.asarray(ref)
        rows.append(
            {
                "k": k,
                "log_w": float(log_w),
                "gap": float(ref.mean() - log_w),
                "gap_se": float(ref.std() * np.sqrt(1 + 1 / references)),
            }
        )
    frame = pd.DataFrame(rows)
    nxt = frame.shift(-1)
    frame["gap_rule"] = frame["gap"] >= nxt["gap"] - nxt["gap_se"]
    return frame


def gap_choice(gap: pd.DataFrame) -> int | None:
    """The smallest k with Gap(k) >= Gap(k + 1) - s(k + 1); None when no k in the range meets it."""
    hits = gap.loc[gap["gap_rule"].fillna(False).astype(bool), "k"]
    return int(hits.iloc[0]) if len(hits) else None


def best_jaccard(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """For each reference segment, the Jaccard overlap with its best matching segment in ``other``."""
    out = []
    for segment in np.unique(reference):
        members = reference == segment
        best = 0.0
        for candidate in np.unique(other[members]):
            hit = other == candidate
            best = max(best, (members & hit).sum() / (members | hit).sum())
        out.append(best)
    return np.asarray(out)


def stability(x: np.ndarray, k: int, fits: int, seed: int, n_init: int) -> dict:
    rng = np.random.default_rng(seed)
    reference = kmeans(x, k, seed, n_init).labels_
    ari, jaccard = [], []
    for i in range(fits):
        rows = rng.integers(0, len(x), len(x))
        refit = kmeans(x[rows], k, seed + 1 + i, n_init)
        assigned = refit.predict(x)
        ari.append(adjusted_rand_score(reference, assigned))
        jaccard.append(best_jaccard(reference, assigned))
    ari = np.asarray(ari)
    jaccard = np.vstack(jaccard).mean(axis=0)
    return {
        "k": k,
        "stability_ari": float(ari.mean()),
        "stability_ari_p05": float(np.quantile(ari, 0.05)),
        "min_segment_jaccard": float(jaccard.min()),
    }


def dbscan_grid(x: np.ndarray, min_samples: int, quantiles: list[float], seed: int):
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(x).kneighbors(x)
    rows, labels = [], {}
    for q in quantiles:
        eps = float(np.quantile(distances[:, -1], q))
        name = f"dbscan_q{q:g}"
        labels[name] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
        rows.append({"method": name, "eps": eps, **quality(x, labels[name], seed)})
    return pd.DataFrame(rows), labels


def hdbscan_grid(x: np.ndarray, sizes: list[int], min_samples: int, seed: int):
    rows, labels = [], {}
    for size in sizes:
        name = f"hdbscan_m{size}"
        labels[name] = HDBSCAN(min_cluster_size=size, min_samples=min_samples, copy=True).fit_predict(x)
        rows.append({"method": name, "min_cluster_size": size, **quality(x, labels[name], seed)})
    return pd.DataFrame(rows), labels


def quintile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Scores 1 to 5 from quintiles of the rank, so that ties do not merge quintiles."""
    rank = values.rank(method="first", ascending=higher_is_better)
    return pd.qcut(rank, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_grid(table: pd.DataFrame) -> pd.Series:
    r = quintile_score(table["recency_days"], higher_is_better=False)
    f = quintile_score(table["frequency"], higher_is_better=True)
    code = r.astype(str) + f.astype(str)
    out = pd.Series(pd.NA, index=table.index, dtype=object)
    for pattern, name in RFM_GRID:
        hit = out.isna() & code.str.fullmatch(pattern)
        out[hit] = name
    if out.isna().any():
        raise ValueError(f"RFM codes without a segment: {sorted(code[out.isna()].unique())}")
    return out
