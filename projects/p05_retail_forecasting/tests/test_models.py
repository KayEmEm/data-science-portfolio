import numpy as np
import pandas as pd
import pytest

from retail_forecast import gbm, reconcile
from retail_forecast.hierarchy import build_hierarchy, store_dept_summing_matrix
from retail_forecast.statistical import ets_forecast, fill_closures, seasonal_naive


def test_seasonal_naive_repeats_the_last_week():
    history = np.arange(21, dtype=float).reshape(1, 21)
    forecast = seasonal_naive(history, horizon=10)
    assert forecast[0, :7].tolist() == list(range(14, 21))
    assert forecast[0, 7:].tolist() == [14, 15, 16]


def test_fill_closures_uses_the_same_weekday():
    history = np.array([[10.0] * 7 + [20.0] * 7 + [30.0] * 7])
    closed = np.zeros(21, dtype=bool)
    closed[9] = True
    assert fill_closures(history, closed)[0, 9] == pytest.approx(20.0)  # mean of days 2 and 16
    closed[:] = False
    closed[2] = True
    assert fill_closures(history, closed)[0, 2] == pytest.approx(20.0)  # only day 9 is available


def test_ets_follows_a_weekly_pattern():
    rng = np.random.default_rng(0)
    pattern = np.array([100, 120, 90, 80, 85, 140, 160], dtype=float)
    history = np.tile(pattern, 30)[None, :] + rng.normal(0, 3, (1, 210))
    point, bounds, residuals = ets_forecast(history, horizon=14)
    assert np.abs(point[0] - np.tile(pattern, 2)).mean() < 10
    lower, upper = bounds[0.8]
    assert (lower <= point).all() and (point <= upper).all()
    assert residuals.shape == (1, 210 - 28)


@pytest.fixture(scope="module")
def summing(tables):
    return store_dept_summing_matrix(tables["series"])[0]


@pytest.mark.parametrize("method", reconcile.METHODS)
def test_reconciled_forecasts_are_coherent(summing, method):
    rng = np.random.default_rng(1)
    base = rng.uniform(10, 50, (summing.shape[0], 28))
    residuals = rng.normal(0, 1, (summing.shape[0], 120))
    out = reconcile.reconcile(base, summing, method, residuals=residuals)
    bottom = out[-summing.shape[1] :]
    assert np.allclose(out, summing @ bottom)


@pytest.mark.parametrize("method", reconcile.METHODS)
def test_coherent_base_forecasts_are_unchanged(summing, method):
    rng = np.random.default_rng(2)
    base = summing @ rng.uniform(10, 50, (summing.shape[1], 28))
    residuals = rng.normal(0, 1, (summing.shape[0], 120))
    assert np.allclose(reconcile.reconcile(base, summing, method, residuals=residuals), base)


def test_shrinkage_intensity_is_a_share():
    rng = np.random.default_rng(3)
    _, lam = reconcile.shrunk_covariance(rng.normal(size=(6, 50)))
    assert 0 <= lam <= 1


def test_lightgbm_features_do_not_read_the_last_28_days(tables):
    sales, prices = tables["sales"], tables["prices"]
    days = np.arange(300, 328)
    changed = sales.copy()
    changed[:, 300:] += 50  # every day from the first feature day on
    before = gbm.feature_frame(sales, prices, tables["series"], tables["calendar"], days)
    after = gbm.feature_frame(changed, prices, tables["series"], tables["calendar"], days)
    pd.testing.assert_frame_equal(before[gbm.FEATURES], after[gbm.FEATURES])


def test_lightgbm_never_reads_sales_after_the_origin(tables):
    sales = tables["sales"]
    args = (tables["prices"], tables["series"], tables["calendar"], 450)
    kwargs = {"train_days": 100, "rounds": 20, "leaves": 7, "learning_rate": 0.1}
    first, _ = gbm.fit_predict(sales, *args, **kwargs)
    future_changed = sales.copy()
    future_changed[:, 450:] = 999
    second, _ = gbm.fit_predict(future_changed, *args, **kwargs)
    assert first.shape == (54, 28)
    assert np.allclose(first, second)
    levels = build_hierarchy(tables["series"])
    assert levels[1].aggregate(first).sum() > 0
