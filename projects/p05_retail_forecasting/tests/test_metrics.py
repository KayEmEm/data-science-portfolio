import numpy as np
import pytest

from retail_forecast.hierarchy import build_hierarchy
from retail_forecast.metrics import Evaluator, coverage, mape, rmsse, scales


def test_scales_skip_leading_zeros():
    history = np.array([[0, 0, 2, 4, 1], [0, 0, 0, 0, 0]])
    # changes from the first nonzero day on: 2->4, 4->1
    assert scales(history)[0] == pytest.approx((4 + 9) / 2)
    assert np.isnan(scales(history)[1])


def test_rmsse_matches_its_definition():
    actual = np.array([[1.0, 2.0, 3.0]])
    forecast = np.array([[1.0, 1.0, 1.0]])
    assert rmsse(actual, forecast, np.array([2.0]))[0] == pytest.approx(np.sqrt((0 + 1 + 4) / 3 / 2))


def test_perfect_forecast_scores_zero_and_weights_sum_to_one(tables):
    levels = build_hierarchy(tables["series"])
    ev = Evaluator(tables["sales"], tables["prices"], levels, 400)
    for weight in ev.weight.values():
        assert weight.sum() == pytest.approx(1.0)
    score = ev.score(tables["sales"][:, 400:428].astype(float))
    assert score["wrmsse"] == pytest.approx(0.0)
    worse = ev.score(tables["sales"][:, 400:428] + 1.0)
    assert worse["wrmsse"] > 0


def test_mape_and_coverage():
    actual = np.array([0.0, 10.0, 20.0])
    assert mape(actual, np.array([5.0, 11.0, 18.0])) == pytest.approx(0.1)
    assert coverage(actual, actual - 1, actual + 1) == 1.0
    assert coverage(actual, actual + 1, actual + 2) == 0.0
