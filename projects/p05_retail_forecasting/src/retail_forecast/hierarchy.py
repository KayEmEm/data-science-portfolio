"""The M5 hierarchy: aggregation matrices from item and store series to every level."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from retail_forecast.config import LEVELS, STORE_DEPT_LEVEL


@dataclass
class Level:
    """One aggregation level: the group names and the matrix that sums bottom series into them."""

    number: int
    keys: pd.DataFrame  # one row per aggregate series, the id columns of the level
    matrix: sparse.csr_matrix  # aggregates x bottom series, entries 0 or 1

    @property
    def names(self) -> list[str]:
        if self.keys.empty or not len(self.keys.columns):
            return ["Total"]
        return self.keys.astype(str).agg("_".join, axis=1).tolist()

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """Sum a bottom level matrix (series x days) into this level (aggregates x days)."""
        return np.asarray(self.matrix @ bottom)


def build_level(series: pd.DataFrame, number: int) -> Level:
    columns = LEVELS[number]
    n = len(series)
    if not columns:
        return Level(number, pd.DataFrame(index=[0]), sparse.csr_matrix(np.ones((1, n))))
    keys = series[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)
    codes = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(series[columns]))
    matrix = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(keys), n))
    return Level(number, keys, matrix)


def build_hierarchy(series: pd.DataFrame) -> dict[int, Level]:
    return {number: build_level(series, number) for number in LEVELS}


def store_dept_frame(series: pd.DataFrame) -> pd.DataFrame:
    """The 70 store x department series with their state and category, in level 9 order."""
    columns = ["state_id", "store_id", "cat_id", "dept_id"]
    frame = series[columns].drop_duplicates().sort_values(LEVELS[STORE_DEPT_LEVEL])
    return frame.reset_index(drop=True)


def store_dept_summing_matrix(series: pd.DataFrame) -> tuple[np.ndarray, list[int], list[str]]:
    """Summing matrix S of levels 1 to 9 over the store x department series.

    Returns S (154 aggregates x 70 level 9 series on the M5 data), the level
    number of each row and its name. Levels 1 to 8 group by columns that are
    fixed within a store x department series, so each of their series is a sum
    of whole level 9 series.
    """
    frame = store_dept_frame(series)
    blocks, owners, names = [], [], []
    for number in range(1, STORE_DEPT_LEVEL + 1):
        level = build_level(frame, number)
        blocks.append(level.matrix.toarray())
        owners += [number] * level.matrix.shape[0]
        names += level.names
    return np.vstack(blocks), owners, names


def bottom_shares(sales: np.ndarray, level: Level, window: int = 28) -> np.ndarray:
    """Share of each bottom series in its aggregate over the last ``window`` days.

    Used to split an aggregate forecast down to the item and store series
    (top down). An aggregate with no sales in the window splits evenly.
    """
    recent = sales[:, -window:].sum(axis=1).astype(float)
    totals = np.asarray(level.matrix @ recent).ravel()
    group = np.asarray(level.matrix.argmax(axis=0)).ravel()
    counts = np.asarray(level.matrix.sum(axis=1)).ravel()
    even = 1.0 / counts[group]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(totals[group] > 0, recent / totals[group], even)
    return share


def disaggregate(forecast: np.ndarray, level: Level, shares: np.ndarray) -> np.ndarray:
    """Split aggregate forecasts (aggregates x horizon) to bottom series by fixed shares."""
    group = np.asarray(level.matrix.argmax(axis=0)).ravel()
    return forecast[group] * shares[:, None]
