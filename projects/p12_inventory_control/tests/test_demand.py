import numpy as np
import pytest

from inventory_rl import demand


def test_fit_recovers_weekday_means_and_dispersion():
    rng = np.random.default_rng(0)
    model = {"weekday_mean": [8.0, 7.0, 5.0, 4.0, 4.0, 5.0, 6.0], "dispersion": 3.0}
    first = np.ones(1, dtype="int64")
    units = demand.draw(model, rng, 1, 7 * 3000, first)[0]
    wday = (np.arange(len(units)) % 7) + 1
    fitted = demand.fit(units, wday)
    np.testing.assert_allclose(fitted["weekday_mean"], model["weekday_mean"], rtol=0.03)
    assert fitted["dispersion"] == pytest.approx(3.0, rel=0.08)


def test_closed_days_are_left_out_of_the_fit():
    units = np.full(70, 10)
    units[::10] = 0
    closed = units == 0
    fitted = demand.fit(units, np.arange(70) % 7 + 1, closed)
    assert fitted["days"] == 63
    np.testing.assert_allclose(fitted["weekday_mean"], 10.0)


def test_poisson_like_data_hit_the_upper_bound():
    rng = np.random.default_rng(1)
    units = rng.poisson(4.0, size=5000)
    fitted = demand.fit(units, np.arange(5000) % 7 + 1)
    assert fitted["dispersion"] > 100


def test_interval_coverage_is_near_its_level():
    rng = np.random.default_rng(2)
    model = {"weekday_mean": [5.0] * 7, "dispersion": 2.0}
    units = demand.draw(model, rng, 1, 20_000, 1)[0]
    wday = np.arange(20_000) % 7 + 1
    # Discrete intervals cover at least their level.
    assert 0.80 <= demand.interval_coverage(model, units, wday, 0.80) <= 0.90


def test_draw_follows_the_weekday_of_day_zero():
    rng = np.random.default_rng(3)
    model = {"weekday_mean": [50.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], "dispersion": 1e4}
    units = demand.draw(model, rng, 2, 14, np.array([1, 3]))
    assert units[0, 0] > 20 and units[0, 7] > 20
    assert units[1, 5] > 20 and units[1, 0] == 0
