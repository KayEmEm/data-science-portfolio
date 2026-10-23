import duckdb
import pandas as pd
import pytest

from taxi_demand.warehouse import analysis_queries, build_scripts, run_query


@pytest.fixture()
def con(built):
    with duckdb.connect(str(built["warehouse"]), read_only=True) as connection:
        yield connection


def test_build_scripts_run_in_numbered_order():
    names = build_scripts()
    assert names == sorted(names)
    assert names[0].startswith("01_")


def test_audit_accounts_for_every_source_row(con, raw_dir):
    source_rows = sum(len(pd.read_parquet(p)) for p in (raw_dir / "trips").glob("*.parquet"))
    audit = con.sql("SELECT * FROM quality_audit").df()
    assert audit["trips"].sum() == source_rows
    assert audit["share"].sum() == pytest.approx(1.0)
    # The synthetic generator plants each of these defects.
    for label in [
        "zero distance",
        "negative amount",
        "unknown zone",
        "pickup outside year",
        "duration over limit",
    ]:
        assert label in set(audit["check_result"])


def test_usable_trips_pass_every_check(con):
    bad = con.sql(
        """
        SELECT count(*) AS n FROM trips
        WHERE issue IS NULL AND (
            distance_mi <= 0 OR duration_min <= 0 OR fare_usd < 0 OR year(pickup_at) <> 2024
            OR pu_zone_id > 263 OR payment_type IN (3, 4, 6))
        """
    ).df()
    assert bad["n"].iloc[0] == 0


def test_fact_table_preserves_usable_trip_totals(con):
    trips, fare = con.execute("SELECT count(*), sum(fare_usd) FROM trips WHERE issue IS NULL").fetchone()
    fact_trips, fact_fare = con.execute("SELECT sum(trips), sum(fare_usd) FROM fact_hourly").fetchone()
    assert fact_trips == trips
    assert fact_fare == pytest.approx(fare)


def test_demand_index_averages_one_per_weekday_and_month(con):
    means = con.sql("SELECT avg(demand_index) AS m FROM daily GROUP BY pickup_month, pickup_dow").df()
    assert means["m"].to_numpy() == pytest.approx(1.0)


def test_every_day_has_weather(con):
    missing = con.execute("SELECT count(*) FROM daily WHERE weather_class = 'no record'").fetchone()[0]
    assert missing == 0


@pytest.mark.parametrize("name", analysis_queries())
def test_analysis_query_returns_rows(con, name):
    assert len(run_query(con, name)) > 0


def test_zone_concentration_ends_at_one(con):
    zones = run_query(con, "a05_zone_concentration")
    assert zones["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert zones["trips"].is_monotonic_decreasing


def test_cohorts_start_at_100(con):
    cohorts = run_query(con, "a08_zone_cohorts")
    january = cohorts[cohorts["month"] == 1]
    assert january["index_january_100"].to_numpy() == pytest.approx(100.0)
    assert set(cohorts["volume_quartile"]) == {1, 2, 3, 4}


def test_marts_are_exported(built):
    for name in ["fact_hourly", "zone_monthly", "daily", "quality_audit", "zones"]:
        assert (built["marts"] / f"{name}.parquet").exists()
