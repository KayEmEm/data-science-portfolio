"""Download the TLC yellow taxi trip files, the zone lookup and NOAA daily weather; check their schemas."""

import argparse
import csv
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from taxi_demand.config import (
    RAW_DIR,
    TRIP_COLUMNS,
    TRIPS_URL,
    WEATHER_COLUMNS,
    WEATHER_STATION,
    WEATHER_URL,
    YEAR,
    ZONE_COLUMNS,
    ZONES_URL,
)


class SchemaError(ValueError):
    """A source file lacks a required column."""


def trip_file(raw_dir: Path, year: int, month: int) -> Path:
    return raw_dir / "trips" / f"yellow_tripdata_{year}-{month:02d}.parquet"


def fetch(url: str, out: Path) -> Path:
    """Download ``url`` to ``out`` unless the file is already present. A partial file is never left behind."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "data-science-portfolio/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as sink:
        shutil.copyfileobj(response, sink)
    partial.replace(out)
    return out


def missing_columns(found: list[str], required: list[str]) -> list[str]:
    """Required columns absent from ``found``. The match ignores case, as DuckDB does."""
    lower = {c.lower() for c in found}
    return [c for c in required if c.lower() not in lower]


def check_trip_file(path: Path) -> int:
    """Check the columns of one monthly trip file and return its row count."""
    meta = pq.ParquetFile(path)
    missing = missing_columns(meta.schema_arrow.names, TRIP_COLUMNS)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {missing}")
    return meta.metadata.num_rows


def check_csv(path: Path, required: list[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    missing = missing_columns([c.strip() for c in header], required)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {missing}")


def download_all(
    raw_dir: Path, year: int, months: list[int], skip_download: bool, allow_missing: bool = False
) -> None:
    zones = raw_dir / "taxi_zone_lookup.csv"
    weather = raw_dir / "weather_central_park.csv"
    if not skip_download:
        fetch(ZONES_URL, zones)
        fetch(WEATHER_URL.format(station=WEATHER_STATION, year=year), weather)
    check_csv(zones, ZONE_COLUMNS)
    check_csv(weather, WEATHER_COLUMNS)

    total = 0
    for month in months:
        path = trip_file(raw_dir, year, month)
        if not skip_download:
            try:
                fetch(TRIPS_URL.format(year=year, month=month), path)
            except urllib.error.HTTPError as error:
                # The TLC host answers 403 or 404 for a month that is not published yet.
                if allow_missing and error.code in (403, 404):
                    print(f"{year}-{month:02d}: not published yet ({error.code}), skipped")
                    continue
                raise SystemExit(
                    f"{year}-{month:02d}: the TLC file is not available ({error.code})"
                ) from error
        rows = check_trip_file(path)
        total += rows
        print(f"{path.name}: {rows:,} rows")
    print(f"{len(months)} monthly files, {total:,} rows in total, under {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--skip-download", action="store_true", help="only check files already placed under --raw-dir"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip months the TLC has not published yet (for scheduled runs)",
    )
    args = parser.parse_args()
    download_all(args.raw_dir, args.year, args.months, args.skip_download, args.allow_missing)


if __name__ == "__main__":
    main()
