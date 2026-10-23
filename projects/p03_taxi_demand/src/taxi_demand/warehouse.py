"""Build the DuckDB warehouse from the raw files and export the small tables the dashboard reads."""

import argparse
import time
from importlib.resources import files
from pathlib import Path

import duckdb
import pandas as pd

from taxi_demand import config

BUILD_SQL = files("taxi_demand.sql")
ANALYSIS_SQL = BUILD_SQL / "analysis"
MART_TABLES = ["zones", "quality_audit", "fact_hourly", "zone_monthly", "daily"]


def build_parameters(year: int) -> dict[str, float]:
    return {
        "year": year,
        "max_duration_min": config.MAX_DURATION_MIN,
        "max_distance_mi": config.MAX_DISTANCE_MI,
        "max_fare_usd": config.MAX_FARE_USD,
        "min_speed_mph": config.MIN_SPEED_MPH,
        "max_speed_mph": config.MAX_SPEED_MPH,
    }


def build_scripts() -> list[str]:
    return sorted(p.name for p in BUILD_SQL.iterdir() if p.name.endswith(".sql"))


def analysis_queries() -> list[str]:
    return sorted(p.name.removesuffix(".sql") for p in ANALYSIS_SQL.iterdir() if p.name.endswith(".sql"))


def register_raw(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Expose the raw files as the views raw_trips, raw_zones and raw_weather.

    The relation API takes the paths as values, so no path is spliced into SQL text.
    union_by_name aligns monthly files whose column order or numeric types differ.
    """
    trip_files = sorted(str(p) for p in (raw_dir / "trips").glob("*.parquet"))
    if not trip_files:
        raise FileNotFoundError(f"no trip files under {raw_dir / 'trips'}; run make data first")
    con.read_parquet(trip_files, union_by_name=True).to_view("raw_trips")
    con.read_csv(str(raw_dir / "taxi_zone_lookup.csv"), header=True).to_view("raw_zones")
    con.read_csv(str(raw_dir / "weather_central_park.csv"), header=True, all_varchar=True).to_view(
        "raw_weather"
    )


def build(raw_dir: Path, warehouse: Path, marts_dir: Path, year: int) -> None:
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    marts_dir.mkdir(parents=True, exist_ok=True)
    params = build_parameters(year)
    with duckdb.connect(str(warehouse)) as con:
        register_raw(con, raw_dir)
        for name in build_scripts():
            started = time.perf_counter()
            sql = (BUILD_SQL / name).read_text(encoding="utf-8")
            used = {k: v for k, v in params.items() if f"${k}" in sql}
            con.execute(sql, used)
            print(f"{name}: {time.perf_counter() - started:.1f} s")
        for table in MART_TABLES:
            con.table(table).write_parquet(str(marts_dir / f"{table}.parquet"))
        kept = con.execute("SELECT trips, share FROM quality_audit WHERE check_result = 'kept'").fetchone()
    print(
        f"warehouse {warehouse}; {kept[0]:,} usable trips ({kept[1]:.1%} of source rows); "
        f"marts in {marts_dir}"
    )


def run_query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Run one query of the analysis library by its file name without suffix."""
    return con.sql((ANALYSIS_SQL / f"{name}.sql").read_text(encoding="utf-8")).df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--warehouse", type=Path, default=config.WAREHOUSE_PATH)
    parser.add_argument("--marts", type=Path, default=config.MARTS_DIR)
    parser.add_argument("--year", type=int, default=config.YEAR)
    args = parser.parse_args()
    build(args.raw_dir, args.warehouse, args.marts, args.year)


if __name__ == "__main__":
    main()
