"""M5 accuracy measures: RMSSE, WRMSSE over the twelve levels, MAPE and interval coverage."""

import numpy as np
import pandas as pd

from retail_forecast.config import HORIZON, LEVELS
from retail_forecast.hierarchy import Level


def scales(history: np.ndarray) -> np.ndarray:
    """Mean squared one day change of each series, from its first nonzero day on.

    This is the RMSSE denominator of the M5 guide. Leading zeros (days before
    an item was stocked) are left out. A series that never sold gets NaN.
    """
    history = np.asarray(history, dtype=float)
    started = np.maximum.accumulate(history > 0, axis=1)
    diffs = np.diff(history, axis=1) ** 2
    valid = started[:, :-1]  # a change counts once the series has started
    counts = valid.sum(axis=1)
    totals = np.where(valid, diffs, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def rmsse(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Root mean squared scaled error per series; NaN where the scale is undefined or zero."""
    mse = np.mean((np.asarray(actual, float) - np.asarray(forecast, float)) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, np.sqrt(mse / scale), np.nan)


def dollar_weights(sales: np.ndarray, prices: np.ndarray, origin: int, level: Level) -> np.ndarray:
    """Share of each series of a level in dollar sales over the 28 days up to the origin."""
    window = slice(origin - HORIZON, origin)
    dollars = (sales[:, window] * np.nan_to_num(prices[:, window])).sum(axis=1)
    totals = np.asarray(level.matrix @ dollars).ravel()
    return totals / totals.sum()


class Evaluator:
    """Scores bottom level forecasts of one window on all twelve M5 levels.

    The weights and scales depend only on the history up to the origin, so they
    are computed once per window and shared by every model.
    """

    def __init__(self, sales: np.ndarray, prices: np.ndarray, levels: dict[int, Level], origin: int):
        self.levels = levels
        self.origin = origin
        history = sales[:, :origin]
        self.actual = {n: lv.aggregate(sales[:, origin : origin + HORIZON]) for n, lv in levels.items()}
        self.scale = {n: scales(lv.aggregate(history)) for n, lv in levels.items()}
        self.weight = {n: dollar_weights(sales, prices, origin, lv) for n, lv in levels.items()}

    def level_rmsse(self, bottom_forecast: np.ndarray) -> dict[int, np.ndarray]:
        out = {}
        for n, level in self.levels.items():
            out[n] = rmsse(self.actual[n], level.aggregate(bottom_forecast), self.scale[n])
        return out

    def score(self, bottom_forecast: np.ndarray) -> pd.Series:
        """WRMSSE per level and overall (the mean of the twelve level scores)."""
        per_level = {}
        for n, values in self.level_rmsse(bottom_forecast).items():
            per_level[n] = float(np.nansum(self.weight[n] * values))
        result = pd.Series({f"level_{n}": v for n, v in per_level.items()})
        result["wrmsse"] = float(np.mean(list(per_level.values())))
        return result


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error over the days with positive actual units."""
    actual = np.asarray(actual, float)
    forecast = np.asarray(forecast, float)
    positive = actual > 0
    return float(np.mean(np.abs(actual[positive] - forecast[positive]) / actual[positive]))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actual values inside their interval."""
    return float(np.mean((actual >= lower) & (actual <= upper)))


def scaled_width(lower: np.ndarray, upper: np.ndarray, scale: np.ndarray) -> float:
    """Mean interval width in units of each series' scale (root of the RMSSE denominator)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        width = (upper - lower) / np.sqrt(scale)[:, None]
    return float(np.nanmean(width))


def level_labels() -> list[str]:
    return [f"level_{n}" for n in LEVELS]
