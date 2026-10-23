"""Place manually downloaded source files under data/raw, identifying each file by its content.

For use when the download hosts are not reachable: the files may carry any name.
Trip files are named by the month most of their pickups fall in.
"""

import argparse
import csv
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from taxi_demand.config import RAW_DIR, TRIP_COLUMNS, WEATHER_COLUMNS, ZONE_COLUMNS
from taxi_demand.data import check_trip_file, missing_columns


def kind_of(path: Path) -> str:
    """'trips', 'zones', 'weather' or 'other'."""
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == b"PAR1":
        names = pq.ParquetFile(path).schema_arrow.names
        return "trips" if not missing_columns(names, TRIP_COLUMNS) else "other"
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = [c.strip() for c in next(csv.reader(handle))]
    except (UnicodeDecodeError, StopIteration, csv.Error):
        return "other"
    if not missing_columns(header, ZONE_COLUMNS):
        return "zones"
    if not missing_columns(header, WEATHER_COLUMNS):
        return "weather"
    return "other"


def trip_month(path: Path) -> tuple[int, int]:
    """The (year, month) that most pickups in the file fall in."""
    column = pq.read_table(path, columns=["tpep_pickup_datetime"]).column(0)
    keys = pc.add(pc.multiply(pc.year(column), 100), pc.month(column))
    top = pc.mode(keys, 1)[0]["mode"].as_py()
    return divmod(top, 100)


def ingest(sources: list[Path], raw_dir: Path) -> dict[str, list[Path]]:
    placed: dict[str, list[Path]] = {"trips": [], "zones": [], "weather": [], "other": []}
    files = [p for s in sources for p in ([s] if s.is_file() else sorted(s.rglob("*"))) if p.is_file()]
    for path in files:
        kind = kind_of(path)
        if kind == "trips":
            year, month = trip_month(path)
            target = raw_dir / "trips" / f"yellow_tripdata_{year}-{month:02d}.parquet"
        elif kind == "zones":
            target = raw_dir / "taxi_zone_lookup.csv"
        elif kind == "weather":
            target = raw_dir / "weather_central_park.csv"
        else:
            placed["other"].append(path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if kind == "trips":
            check_trip_file(target)
        placed[kind].append(target)
        print(f"{path} -> {target}")
    return placed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+", help="files or folders to scan")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    placed = ingest(args.sources, args.raw_dir)
    print({k: len(v) for k, v in placed.items()})


if __name__ == "__main__":
    main()
