from datetime import date

import pytest

from taxi_demand.dashboard_data import (
    Filters,
    apply_filters,
    hour_weekday,
    kpis,
    load_marts,
    previous_period,
    relative_change,
    zone_ranking,
)


@pytest.fixture(scope="module")
def marts(built):
    return load_marts(built["marts"])


def all_filters(fact, start=date(2024, 1, 1), end=date(2024, 3, 31)):
    return Filters(start, end, tuple(fact["pickup_borough"].unique()), tuple(fact["payment"].unique()))


def test_unfiltered_kpis_match_the_fact_table(marts):
    fact = marts["fact_hourly"]
    values = kpis(apply_filters(fact, all_filters(fact)))
    assert values["trips"] == fact["trips"].sum()
    card = fact[fact["payment"] == "Credit card"]
    assert values["card_tip_rate"] == pytest.approx(card["tip_usd"].sum() / card["fare_usd"].sum())


def test_filters_partition_the_trips(marts):
    fact = marts["fact_hourly"]
    full = all_filters(fact)
    total = sum(
        kpis(apply_filters(fact, Filters(full.start, full.end, (b,), full.payments)))["trips"]
        for b in full.boroughs
    )
    assert total == fact["trips"].sum()


def test_empty_selection_gives_nan_ratios(marts):
    fact = marts["fact_hourly"]
    values = kpis(apply_filters(fact, Filters(date(2024, 1, 1), date(2024, 1, 31), (), ())))
    assert values["trips"] == 0
    assert values["mean_fare_usd"] != values["mean_fare_usd"]


def test_previous_period_has_the_same_length():
    current = Filters(date(2024, 3, 1), date(2024, 3, 31), ("Manhattan",), ("Cash",))
    prior = previous_period(current)
    assert prior.end == date(2024, 2, 29)
    assert (prior.end - prior.start) == (current.end - current.start)


def test_relative_change_handles_missing_prior():
    assert relative_change(110, 100) == pytest.approx(0.10)
    assert relative_change(110, 0) != relative_change(110, 0)


def test_hour_weekday_grid_shape(marts):
    grid = hour_weekday(marts["fact_hourly"])
    assert grid.shape == (7, 24)
    assert list(grid.index) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def test_zone_ranking_respects_borough_filter(marts):
    fact = marts["fact_hourly"]
    filters = Filters(date(2024, 1, 1), date(2024, 3, 31), ("Brooklyn",), tuple(fact["payment"].unique()))
    ranking = zone_ranking(marts["zone_monthly"], filters, top=5)
    assert len(ranking) <= 5
    assert set(ranking["borough"]) == {"Brooklyn"}
