"""Popularity baselines: the same list for every user, minus the movies the user has rated."""

import numpy as np

from movie_recs.interactions import Stage
from movie_recs.metrics import top_n


def popularity_scores(stage: Stage, days: int | None) -> np.ndarray:
    """Number of ratings per catalog item, over the whole history or its last ``days`` days."""
    history = stage.history if days is None else stage.days_before_cutoff(days)
    counts = history["item"].value_counts()
    scores = counts.reindex(stage.items, fill_value=0).to_numpy(dtype=np.float32)
    # Ties are broken by the count over the whole history, so that old favourites rank above unknowns.
    total = stage.history["item"].value_counts().reindex(stage.items, fill_value=0).to_numpy(np.float32)
    return scores + total / (total.max() + 1.0)


def recommend(stage: Stage, users: np.ndarray, n: int, days: int | None) -> tuple[np.ndarray, np.ndarray]:
    scores = popularity_scores(stage, days)
    return top_n(
        lambda rows: np.broadcast_to(scores, (len(rows), len(scores))),
        stage.user_rows(users),
        stage.matrix,
        stage.items,
        n,
    )
