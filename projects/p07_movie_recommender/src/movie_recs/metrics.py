"""Top N selection without seen movies, and the offline measures of a list of recommendations.

* recall at k: the share of a user's liked movies in the window that the top k contains;
* NDCG at k: the same hits, discounted by rank, divided by the best possible score;
* coverage: the share of the catalog that appears in at least one top k list;
* novelty: the mean self information, -log2 of the share of history users who
  rated a recommended movie. Higher means less popular movies.

Recall and NDCG are averaged over users, so a heavy user counts as much as a light one;
their 95 percent intervals treat users as independent draws.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

PAD = -1  # item id that fills a list shorter than n


def top_n(
    scores: Callable[[np.ndarray], np.ndarray],
    rows: np.ndarray,
    seen: sp.csr_matrix,
    items: np.ndarray,
    n: int,
    batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Best n items per user row by ``scores(rows) -> [len(rows), n_items]``, seen items excluded.

    Returns item ids and scores, each [len(rows), n], best first.
    """
    n = min(n, len(items))
    out_items = np.full((len(rows), n), PAD, dtype=np.int64)
    out_scores = np.full((len(rows), n), -np.inf, dtype=np.float32)
    for lo in range(0, len(rows), batch):
        chunk = rows[lo : lo + batch]
        dense = np.array(scores(chunk), dtype=np.float32, copy=True)
        sub = seen[chunk]
        dense[np.repeat(np.arange(len(chunk)), np.diff(sub.indptr)), sub.indices] = -np.inf
        part = np.argpartition(-dense, n - 1, axis=1)[:, :n]
        part_scores = np.take_along_axis(dense, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        best = np.take_along_axis(part, order, axis=1)
        best_scores = np.take_along_axis(part_scores, order, axis=1)
        valid = np.isfinite(best_scores)
        out_items[lo : lo + len(chunk)] = np.where(valid, items[best], PAD)
        out_scores[lo : lo + len(chunk)] = best_scores
    return out_items, out_scores


def hit_matrix(users: np.ndarray, recs: np.ndarray, targets: pd.DataFrame) -> np.ndarray:
    """[n_users, k] booleans: is the recommended item one the user liked in the window."""
    key = pd.MultiIndex.from_arrays([targets["user"].to_numpy(), targets["item"].to_numpy()])
    flat = pd.MultiIndex.from_arrays([np.repeat(users, recs.shape[1]), recs.ravel()])
    return flat.isin(key).reshape(recs.shape)


def per_user(users: np.ndarray, recs: np.ndarray, targets: pd.DataFrame, k: int) -> pd.DataFrame:
    """Recall and NDCG at k for each user."""
    recs = recs[:, :k]
    k = recs.shape[1]  # a list may be shorter than k when few items qualify
    hits = hit_matrix(users, recs, targets)
    n_liked = targets.groupby("user").size().reindex(users, fill_value=0).to_numpy()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.cumsum(discounts)[np.clip(np.minimum(n_liked, k) - 1, 0, None)]
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(n_liked > 0, hits.sum(axis=1) / n_liked, np.nan)
        ndcg = np.where(n_liked > 0, dcg / ideal, np.nan)
    return pd.DataFrame(
        {"user": users, "n_liked": n_liked, "hits": hits.sum(axis=1), "recall": recall, "ndcg": ndcg}
    )


def novelty(recs: np.ndarray, popularity: pd.Series, n_users: int) -> float:
    """Mean -log2 share of history users who rated each recommended item."""
    flat = recs.ravel()
    flat = flat[flat != PAD]
    counts = popularity.reindex(flat, fill_value=0).to_numpy()
    share = np.maximum(counts, 1) / n_users
    return float(np.mean(-np.log2(share)))


def half_width(values: pd.Series) -> float:
    """Half width of a normal 95 percent interval for the mean over users."""
    values = values.dropna()
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def summarize(
    users: np.ndarray,
    recs: np.ndarray,
    targets: pd.DataFrame,
    popularity: pd.Series,
    n_history_users: int,
    n_catalog: int,
    k: int,
) -> dict[str, float]:
    """Recall, NDCG, hit rate, coverage and novelty at k over the given users."""
    scores = per_user(users, recs, targets, k)
    top = recs[:, :k]
    distinct = np.unique(top[top != PAD])
    return {
        "users": int(len(users)),
        f"recall@{k}": float(scores["recall"].mean()),
        f"recall@{k}_ci95": half_width(scores["recall"]),
        f"ndcg@{k}": float(scores["ndcg"].mean()),
        f"ndcg@{k}_ci95": half_width(scores["ndcg"]),
        f"hit_rate@{k}": float((scores["hits"] > 0).mean()),
        f"coverage@{k}": float(len(distinct) / n_catalog),
        f"novelty@{k}": novelty(top, popularity, n_history_users),
    }
