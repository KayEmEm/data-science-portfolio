"""Data for the demo page: the history of one test user and the top 10 of every model."""

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recs.config import K
from movie_recs.evaluate import MODELS
from movie_recs.interactions import Stage
from movie_recs.metrics import PAD
from movie_recs.train import load_lists, stage_dir


def load_lists_by_model(artifacts: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lists = {}
    for model in MODELS:
        users, items, _ = load_lists(stage_dir(artifacts, "test") / "lists" / f"{model}.npz")
        lists[model] = (users, items[:, :K])
    return lists


def user_view(stage: Stage, movies: pd.DataFrame, lists: dict, user: int, n_history: int = 15) -> dict:
    """``history``: the user's last rated movies before the test window.
    ``recommendations``: one column per model, one row per rank; a liked movie is marked with a star.
    """
    titles = movies.set_index("item")["title"]
    history = (
        stage.history.loc[stage.history["user"] == user].sort_values("ts", ascending=False).head(n_history)
    )
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(history["ts"], unit="s").dt.date,
            "movie": titles.reindex(history["item"]).to_numpy(),
            "rating": history["rating"].to_numpy(),
        }
    )
    liked = set(stage.targets.loc[stage.targets["user"] == user, "item"])
    columns = {}
    for model, (users, items) in lists.items():
        row = np.searchsorted(users, user)
        if row >= len(users) or users[row] != user:
            continue
        columns[model] = [
            "" if i == PAD else ("* " if i in liked else "") + str(titles.get(i, i)) for i in items[row]
        ]
    recommendations = pd.DataFrame(columns, index=pd.RangeIndex(1, K + 1, name="rank"))
    later = titles.reindex(sorted(liked)).dropna().tolist()
    return {"history": history, "recommendations": recommendations, "liked_in_window": later}
