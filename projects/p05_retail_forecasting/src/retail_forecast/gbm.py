"""A global LightGBM model on all item and store series.

Every feature of day ``t`` uses sales up to day ``t - 28`` at the latest, so
one model forecasts all 28 days of a window without feeding forecasts back
in. Prices and the calendar are known in advance in M5 and enter for the day
itself.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.config import HORIZON, SEED

LAGS = (28, 35, 42, 49, 56)
WINDOWS = (7, 28, 56, 112)
CATEGORICAL = ["item", "dept", "cat", "store", "state", "event_name", "event_type"]
EARLIEST = HORIZON + max(WINDOWS)  # history a feature needs before the first training day


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values, sort=True)[0].astype("int32")


def _lagged_mean(cumsum: np.ndarray, end: np.ndarray, window: int) -> np.ndarray:
    """Mean of days [end - window + 1, end] (0 based, inclusive) from a cumulative sum with a zero column."""
    return (cumsum[:, end + 1] - cumsum[:, end + 1 - window]) / window


def feature_frame(
    sales: np.ndarray,
    prices: np.ndarray,
    series: pd.DataFrame,
    calendar: pd.DataFrame,
    days: np.ndarray,
) -> pd.DataFrame:
    """Features for every series on the given 0 based day columns, one row per series and day.

    Rows before an item's first price (not yet stocked) are dropped. The frame
    keeps ``row`` (series index), ``day`` and ``units`` (NaN beyond the data).
    """
    n_series, n_days = sales.shape
    if days.min() < EARLIEST:
        raise ValueError(f"the first feature day needs {EARLIEST} days of history")
    x = sales.astype("float32")
    cumsum = np.concatenate(
        [np.zeros((n_series, 1), "float64"), np.cumsum(x, axis=1, dtype="float64")], axis=1
    )
    sq_cumsum = np.concatenate([np.zeros((n_series, 1)), np.cumsum(x.astype("float64") ** 2, axis=1)], axis=1)
    cols: dict[str, np.ndarray] = {}
    for lag in LAGS:
        cols[f"lag_{lag}"] = x[:, days - lag]
    end = days - HORIZON
    for window in WINDOWS:
        cols[f"mean_{window}"] = _lagged_mean(cumsum, end, window).astype("float32")
    mean28 = _lagged_mean(cumsum, end, 28)
    var28 = _lagged_mean(sq_cumsum, end, 28) - mean28**2
    cols["std_28"] = np.sqrt(np.maximum(var28, 0)).astype("float32")

    price = prices[:, days]
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["price"] = price
        cols["price_rel_mean"] = price / np.nanmean(prices, axis=1, keepdims=True)
        cols["price_rel_max"] = price / np.nanmax(prices, axis=1, keepdims=True)
        cols["price_change_7"] = price / prices[:, days - 7] - 1
    released = np.argmax(~np.isnan(prices), axis=1)
    cols["days_on_sale"] = (days[None, :] - released[:, None]).astype("float32")

    shape = (n_series, len(days))
    frame = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    frame.insert(0, "day", np.tile(days, n_series).astype("int32"))
    frame.insert(0, "row", np.repeat(np.arange(n_series, dtype="int32"), len(days)))
    units = np.full(shape, np.nan, dtype="float32")
    inside = days < n_days
    units[:, inside] = x[:, days[inside]]
    frame["units"] = units.reshape(-1)

    for name, column in (("item", "item_id"), ("dept", "dept_id"), ("cat", "cat_id")):
        frame[name] = np.repeat(_codes(series[column]), len(days))
    frame["store"] = np.repeat(_codes(series["store_id"]), len(days))
    state = series["state_id"].to_numpy()
    frame["state"] = np.repeat(_codes(series["state_id"]), len(days))

    cal = calendar.iloc[days]
    dates = pd.DatetimeIndex(cal["date"])
    frame["wday"] = np.tile(cal["wday"].to_numpy("int8"), n_series)
    frame["month"] = np.tile(cal["month"].to_numpy("int8"), n_series)
    frame["mday"] = np.tile(dates.day.to_numpy("int8"), n_series)
    frame["event_name"] = np.tile(_codes(calendar["event_name_1"].fillna(""))[days], n_series)
    frame["event_type"] = np.tile(_codes(calendar["event_type_1"].fillna(""))[days], n_series)
    snap = np.zeros(shape, dtype="int8")
    for code in np.unique(state):
        snap[state == code] = cal[f"snap_{code}"].to_numpy("int8")
    frame["snap"] = snap.reshape(-1)
    return frame[~np.isnan(frame["price"].to_numpy())].reset_index(drop=True)


FEATURES = [
    *[f"lag_{lag}" for lag in LAGS],
    *[f"mean_{w}" for w in WINDOWS],
    "std_28",
    "price",
    "price_rel_mean",
    "price_rel_max",
    "price_change_7",
    "days_on_sale",
    *CATEGORICAL[:5],
    "wday",
    "month",
    "mday",
    "event_name",
    "event_type",
    "snap",
]


def params(leaves: int, learning_rate: float, seed: int = SEED, threads: int = 0) -> dict:
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "learning_rate": learning_rate,
        "num_leaves": leaves,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "seed": seed,
        "deterministic": True,
        "num_threads": threads,
        "verbosity": -1,
    }


def fit_predict(
    sales: np.ndarray,
    prices: np.ndarray,
    series: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: int,
    train_days: int,
    rounds: int,
    leaves: int,
    learning_rate: float,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, lgb.Booster]:
    """Train on the ``train_days`` days up to ``origin`` and forecast the next ``horizon`` days.

    ``origin`` is the number of observed days; sales after it are never read.
    Returns the forecasts (series x horizon) and the booster.
    """
    observed = sales[:, :origin]
    first = max(EARLIEST, origin - train_days)
    future = np.arange(origin, origin + horizon)
    padded = np.concatenate([observed, np.zeros((observed.shape[0], horizon), observed.dtype)], axis=1)
    train = feature_frame(observed, prices, series, calendar, np.arange(first, origin))
    dataset = lgb.Dataset(
        train[FEATURES], label=train["units"], categorical_feature=CATEGORICAL[:5] + CATEGORICAL[5:]
    )
    del train
    booster = lgb.train(params(leaves, learning_rate), dataset, num_boost_round=rounds)
    test = feature_frame(padded, prices, series, calendar, future)
    forecast = np.zeros((sales.shape[0], horizon), dtype="float64")
    forecast[test["row"], test["day"] - origin] = booster.predict(test[FEATURES])
    return np.maximum(forecast, 0), booster
