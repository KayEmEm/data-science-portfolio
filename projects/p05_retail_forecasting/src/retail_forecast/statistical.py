"""Baselines and statistical models: seasonal naive, ETS and Prophet.

Every function forecasts ``horizon`` days after the last column of ``history``
(series x days). Interval functions return, per coverage level, the lower and
upper bounds with the same shape as the point forecast.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from retail_forecast.config import HORIZON

Intervals = dict[float, tuple[np.ndarray, np.ndarray]]


def seasonal_naive(history: np.ndarray, horizon: int = HORIZON, period: int = 7) -> np.ndarray:
    """Repeat the last observed week."""
    last = history[:, -period:]
    reps = -(-horizon // period)
    return np.tile(last, reps)[:, :horizon].astype(float)


def fill_closures(history: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """Replace closed days by the mean of the same weekday one week before and after.

    At the edges, the one available neighbour is used.
    """
    out = history.astype(float).copy()
    n = out.shape[1]
    for day in np.flatnonzero(closed):
        neighbours = [d for d in (day - 7, day + 7) if 0 <= d < n and not closed[d]]
        if neighbours:
            out[:, day] = out[:, neighbours].mean(axis=1)
    return out


def ets_fit(y: np.ndarray):
    """Additive error, damped additive trend and additive weekly season.

    Fitted on log1p units so that the weekly pattern scales with the level and
    forecasts stay positive.
    """
    model = ETSModel(
        pd.Series(np.log1p(y.astype(float))),
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=200)


def ets_forecast(
    history: np.ndarray, horizon: int = HORIZON, coverages: tuple[float, ...] = (0.8, 0.95)
) -> tuple[np.ndarray, Intervals, np.ndarray]:
    """ETS per series: point forecasts, intervals and in sample one step residuals (in units)."""
    n = history.shape[0]
    point = np.zeros((n, horizon))
    bounds = {c: (np.zeros((n, horizon)), np.zeros((n, horizon))) for c in coverages}
    residuals = np.zeros(history.shape, dtype=float)
    for i, y in enumerate(history):
        fit = ets_fit(y)
        pred = fit.get_prediction(start=len(y), end=len(y) + horizon - 1)
        # The mean of a log normal back transform is close to expm1 of the log
        # mean at these volumes; the median back transform is used for points.
        point[i] = np.expm1(np.asarray(pred.predicted_mean))
        for c in coverages:
            frame = pred.summary_frame(alpha=1 - c)
            bounds[c][0][i] = np.expm1(frame["pi_lower"].to_numpy())
            bounds[c][1][i] = np.expm1(frame["pi_upper"].to_numpy())
        residuals[i] = y - np.expm1(np.asarray(fit.fittedvalues))
    # The first weeks of each fit hold the initialization; they are left out of the residuals.
    residuals = residuals[:, 28:]
    point = np.maximum(point, 0)
    bounds = {c: (np.maximum(lo, 0), hi) for c, (lo, hi) in bounds.items()}
    return point, bounds, residuals


def event_holidays(calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar events (sporting, cultural, national, religious) as Prophet holidays."""
    frames = []
    for name in ("event_name_1", "event_name_2"):
        events = calendar.loc[calendar[name].notna(), ["date", name]]
        frames.append(events.rename(columns={"date": "ds", name: "holiday"}))
    holidays = pd.concat(frames).drop_duplicates()
    holidays["lower_window"] = 0
    holidays["upper_window"] = 0
    return holidays.reset_index(drop=True)


def prophet_forecast(
    history: np.ndarray,
    dates: pd.DatetimeIndex,
    calendar: pd.DataFrame,
    horizon: int = HORIZON,
    coverages: tuple[float, ...] = (0.8, 0.95),
    samples: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, Intervals]:
    """Prophet per series with weekly and yearly seasonality and the calendar events."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    holidays = event_holidays(calendar)
    future = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)})
    n = history.shape[0]
    point = np.zeros((n, horizon))
    bounds = {c: (np.zeros((n, horizon)), np.zeros((n, horizon))) for c in coverages}
    for i, y in enumerate(history):
        model = Prophet(
            holidays=holidays,
            weekly_seasonality=True,
            yearly_seasonality=len(y) >= 365,
            daily_seasonality=False,
            uncertainty_samples=samples,
        )
        np.random.seed(seed)
        model.fit(pd.DataFrame({"ds": dates, "y": np.log1p(y.astype(float))}))
        draws = np.expm1(model.predictive_samples(future)["yhat"])  # horizon x samples
        point[i] = np.median(draws, axis=1)
        for c in coverages:
            tail = (1 - c) / 2
            bounds[c][0][i] = np.quantile(draws, tail, axis=1)
            bounds[c][1][i] = np.quantile(draws, 1 - tail, axis=1)
    point = np.maximum(point, 0)
    bounds = {c: (np.maximum(lo, 0), np.maximum(hi, 0)) for c, (lo, hi) in bounds.items()}
    return point, bounds
