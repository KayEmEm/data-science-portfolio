"""The two evaluation stages and the user by item matrix that every model is fitted on.

A stage has a cutoff time. The history is every rating before the cutoff; the
targets are the ratings of 4 or more in the window after it. The validation
stage ends where the test stage starts, so the test window is never seen while
models and settings are chosen.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from movie_recs.config import POSITIVE_RATING

STAGES = ("validation", "test")


@dataclass
class Stage:
    name: str
    start: int  # first second of the window; the history ends just before it
    end: int  # last second of the window
    history: pd.DataFrame  # user, item, rating, ts
    window: pd.DataFrame  # every rating in the window
    users: np.ndarray = field(init=False)  # user ids of the history, row order of the matrix
    items: np.ndarray = field(init=False)  # item ids of the history (the catalog), column order
    matrix: sp.csr_matrix = field(init=False)  # 1 where the user rated the item before the cutoff

    def __post_init__(self) -> None:
        self.users = np.unique(self.history["user"].to_numpy())
        self.items = np.unique(self.history["item"].to_numpy())
        rows = np.searchsorted(self.users, self.history["user"].to_numpy())
        cols = np.searchsorted(self.items, self.history["item"].to_numpy())
        self.matrix = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(self.users), len(self.items))
        )

    @property
    def targets(self) -> pd.DataFrame:
        """The movies each user liked in the window."""
        return self.window.loc[self.window["rating"] >= POSITIVE_RATING, ["user", "item"]]

    def eval_users(self) -> np.ndarray:
        """Users with a liked movie in the window and at least one rating before it."""
        users = np.unique(self.targets["user"].to_numpy())
        return users[np.isin(users, self.users)]

    def new_users(self) -> np.ndarray:
        """Users with a liked movie in the window and no rating before it."""
        users = np.unique(self.targets["user"].to_numpy())
        return users[~np.isin(users, self.users)]

    def user_rows(self, users: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(self.users, users)
        if (rows >= len(self.users)).any() or (
            self.users[np.minimum(rows, len(self.users) - 1)] != users
        ).any():
            raise KeyError("users without history")
        return rows

    def days_before_cutoff(self, days: int) -> pd.DataFrame:
        return self.history.loc[self.history["ts"] >= self.start - days * 86_400]


def make_stage(ratings: pd.DataFrame, split: dict, name: str) -> Stage:
    if name == "validation":
        start, end = split["validation_start"], split["test_start"] - 1
    elif name == "test":
        start, end = split["test_start"], split["end"]
    else:
        raise ValueError(f"unknown stage {name}")
    ts = ratings["ts"].to_numpy()
    history = ratings.loc[ts < start].reset_index(drop=True)
    window = ratings.loc[(ts >= start) & (ts <= end)].reset_index(drop=True)
    return Stage(name=name, start=start, end=end, history=history, window=window)


def split_users(users: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of the validation users: one trains the ranking model, the other compares models."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(users)) < 0.5
    return np.sort(users[mask]), np.sort(users[~mask])
