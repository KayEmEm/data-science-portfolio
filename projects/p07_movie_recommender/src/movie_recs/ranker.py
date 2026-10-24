"""Second stage: a LightGBM LambdaRank model that reorders the union of the ALS and two tower candidates.

The ranking model is trained on one half of the validation users (labels:
the movies they liked in the validation window) and compared with the
retrieval models on the other half. For the test window it is refit on all
validation users and applied to candidates from retrieval models refit on
the longer history. Every feature is computed from the history before the
cutoff of the stage.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from movie_recs.config import GENRES, N_CANDIDATES, RECENT_DAYS, RankerSettings
from movie_recs.interactions import Stage
from movie_recs.metrics import PAD, hit_matrix

FEATURES = [
    "als_score",
    "als_rank",
    "tt_score",
    "tt_rank",
    "in_both",
    "item_log_ratings",
    "item_log_recent",
    "item_mean_rating",
    "item_days_listed",
    "item_year",
    "user_log_ratings",
    "user_mean_rating",
    "user_days_active",
    "user_days_idle",
    "genre_affinity",
    "rating_gap",
]


def long_candidates(users: np.ndarray, items: np.ndarray, scores: np.ndarray, prefix: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "user": np.repeat(users, items.shape[1]),
            "item": items.ravel(),
            f"{prefix}_score": scores.ravel(),
            f"{prefix}_rank": np.tile(np.arange(1, items.shape[1] + 1), len(users)),
        }
    )
    return frame.loc[frame["item"] != PAD]


def candidates(users: np.ndarray, als: tuple, two_tower: tuple) -> pd.DataFrame:
    """Union of both candidate lists; a missing rank is set to one past the list length."""
    a = long_candidates(users, *als, "als")
    t = long_candidates(users, *two_tower, "tt")
    frame = a.merge(t, on=["user", "item"], how="outer")
    frame["in_both"] = (frame["als_rank"].notna() & frame["tt_rank"].notna()).astype("int8")
    for prefix in ("als", "tt"):
        frame[f"{prefix}_rank"] = frame[f"{prefix}_rank"].fillna(N_CANDIDATES + 1)
    return frame.sort_values(["user", "item"]).reset_index(drop=True)


def item_table(stage: Stage, movies: pd.DataFrame) -> pd.DataFrame:
    history = stage.history
    groups = history.groupby("item")
    recent = stage.days_before_cutoff(RECENT_DAYS)["item"].value_counts()
    table = pd.DataFrame(
        {
            "item_log_ratings": np.log1p(groups.size()),
            "item_mean_rating": groups["rating"].mean(),
            "item_days_listed": (stage.start - groups["ts"].min()) / 86_400,
        }
    )
    table["item_log_recent"] = np.log1p(recent.reindex(table.index, fill_value=0))
    meta = movies.set_index("item")
    table["item_year"] = meta["year"].reindex(table.index)
    return table


def user_table(stage: Stage, movies: pd.DataFrame, users: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Activity and mean genre profile of the given users."""
    history = stage.history.loc[stage.history["user"].isin(users)]
    groups = history.groupby("user")
    table = pd.DataFrame(
        {
            "user_log_ratings": np.log1p(groups.size()),
            "user_mean_rating": groups["rating"].mean(),
            "user_days_active": (stage.start - groups["ts"].min()) / 86_400,
            "user_days_idle": (stage.start - groups["ts"].max()) / 86_400,
        }
    )
    genres = movies.set_index("item")[GENRES].astype(np.float32)
    profile = genres.reindex(history["item"]).to_numpy()
    profile = pd.DataFrame(profile, columns=GENRES).groupby(history["user"].to_numpy()).mean()
    return table, profile


def features(stage: Stage, movies: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """The ranking features of every candidate pair in ``frame``."""
    items = item_table(stage, movies)
    users, profile = user_table(stage, movies, frame["user"].unique())
    out = frame.join(items, on="item").join(users, on="user")
    genres = movies.set_index("item")[GENRES].astype(np.float32)
    u = profile.reindex(out["user"]).to_numpy()
    v = genres.reindex(out["item"]).to_numpy()
    norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["genre_affinity"] = np.where(norm > 0, (u * v).sum(axis=1) / norm, 0.0)
    out["rating_gap"] = out["item_mean_rating"] - out["user_mean_rating"]
    return out


def labels(frame: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    return (
        hit_matrix(frame["user"].to_numpy(), frame["item"].to_numpy()[:, None], targets).ravel().astype(int)
    )


def fit(frame: pd.DataFrame, y: np.ndarray, settings: RankerSettings, seed: int) -> lgb.LGBMRanker:
    """``frame`` is sorted by user, so the groups are runs of equal users."""
    groups = frame.groupby("user", sort=False).size().to_numpy()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=settings.n_estimators,
        learning_rate=settings.learning_rate,
        num_leaves=settings.num_leaves,
        min_child_samples=settings.min_child_samples,
        subsample=settings.subsample,
        subsample_freq=1,
        colsample_bytree=settings.colsample_bytree,
        lambdarank_truncation_level=20,
        random_state=seed,
        verbose=-1,
    )
    model.fit(frame[FEATURES], y, group=groups)
    return model


def rank(
    model: lgb.LGBMRanker, frame: pd.DataFrame, users: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top n per user by the ranking model score, in the order of ``users``."""
    scored = frame[["user", "item"]].assign(score=model.predict(frame[FEATURES]))
    scored = scored.sort_values(["user", "score"], ascending=[True, False], kind="stable")
    scored["slot"] = scored.groupby("user").cumcount()
    scored = scored.loc[scored["slot"] < n]
    rows = pd.Index(users).get_indexer(scored["user"])
    items = np.full((len(users), n), PAD, dtype=np.int64)
    scores = np.full((len(users), n), -np.inf, dtype=np.float32)
    items[rows, scored["slot"].to_numpy()] = scored["item"].to_numpy()
    scores[rows, scored["slot"].to_numpy()] = scored["score"].to_numpy()
    return items, scores
