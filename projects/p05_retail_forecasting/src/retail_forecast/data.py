"""Download the M5 archive, check its layout and store wide Parquet tables.

Sales and prices are stored wide: one row per item and store, one column per
day (``d_1`` to ``d_1969``). Every model in this project works on these
matrices; the long demand tables per hierarchy level are written for reuse by
other projects (see ``retail_forecast.demand``).
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample
from portfolio_common.config import parse_args

from retail_forecast.config import (
    CALENDAR_PATH,
    DATA_DIR,
    EXPECTED_DAYS,
    EXPECTED_SERIES,
    ID_COLUMNS,
    OFFICIAL_WEIGHTS_PATH,
    PRICES_PATH,
    RAW_DIR,
    RAW_NAME,
    SALES_PATH,
    SERIES_PATH,
    SOURCE_FILES,
    SOURCE_SHA256,
    SOURCE_URL,
)

CALENDAR_COLUMNS = [
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch the zip (50 MB), unless present, and check that it is the pinned file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_NAME
    if not target.exists():
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(SOURCE_URL, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(target)
    found = sha256(target)
    if found != SOURCE_SHA256:
        raise SchemaError(f"{target} has SHA256 {found}, expected {SOURCE_SHA256}")
    return target


def day_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("d_")]


def read_archive(path: Path) -> dict[str, pd.DataFrame]:
    """Read the five files the project needs from the zip."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(SOURCE_FILES.values()) - names)
        if missing:
            raise SchemaError(f"archive lacks {missing}")
        tables = {}
        for key, name in SOURCE_FILES.items():
            with archive.open(name) as handle:
                tables[key] = pd.read_csv(handle)
    return tables


def check_sales(train: pd.DataFrame, test: pd.DataFrame, expected_series: int | None) -> None:
    for name, frame in (("train", train), ("test", test)):
        missing = [c for c in ID_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"{name} sales lack {missing}")
        if expected_series is not None and len(frame) != expected_series:
            raise SchemaError(f"{name} sales: expected {expected_series} series, found {len(frame)}")
        if frame.duplicated(["item_id", "store_id"]).any():
            raise SchemaError(f"{name} sales repeat an item and store pair")
        days = frame[day_columns(frame)]
        if days.isna().any().any():
            raise SchemaError(f"{name} sales have missing values")
        if (days < 0).any().any():
            raise SchemaError(f"{name} sales have negative units")
    if not train[ID_COLUMNS].equals(test[ID_COLUMNS]):
        raise SchemaError("train and test sales list the series in a different order")
    numbers = [int(c[2:]) for c in day_columns(train) + day_columns(test)]
    if numbers != list(range(1, len(numbers) + 1)):
        raise SchemaError("day columns are not d_1, d_2, ... without gaps")
    hierarchy = train[ID_COLUMNS]
    for child, parent in (("item_id", "dept_id"), ("dept_id", "cat_id"), ("store_id", "state_id")):
        if (hierarchy.groupby(child)[parent].nunique() > 1).any():
            raise SchemaError(f"a {child} belongs to more than one {parent}")


def check_calendar(calendar: pd.DataFrame, n_days: int) -> None:
    missing = [c for c in CALENDAR_COLUMNS if c not in calendar.columns]
    if missing:
        raise SchemaError(f"calendar lacks {missing}")
    if len(calendar) < n_days:
        raise SchemaError(f"calendar has {len(calendar)} days, sales have {n_days}")
    dates = pd.to_datetime(calendar["date"])
    if not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        raise SchemaError("calendar dates are not consecutive")


def normalize(
    tables: dict[str, pd.DataFrame], expected_series: int | None = EXPECTED_SERIES
) -> dict[str, pd.DataFrame]:
    """Check every source table and return the processed tables."""
    train, test = tables["train"], tables["test"]
    check_sales(train, test, expected_series)
    series = train[ID_COLUMNS].reset_index(drop=True)
    sales = pd.concat([train[day_columns(train)], test[day_columns(test)]], axis=1).astype("int32")
    n_days = sales.shape[1]
    if expected_series is not None and n_days != EXPECTED_DAYS:
        raise SchemaError(f"expected {EXPECTED_DAYS} days, found {n_days}")

    calendar = tables["calendar"]
    check_calendar(calendar, n_days)
    calendar = calendar[CALENDAR_COLUMNS].iloc[:n_days].copy()
    calendar.insert(0, "d", np.arange(1, n_days + 1))
    calendar["date"] = pd.to_datetime(calendar["date"])

    prices = tables["prices"]
    if (prices["sell_price"] <= 0).any():
        raise SchemaError("prices must be positive")
    prices = daily_prices(prices, series, calendar)
    if ((sales.to_numpy() > 0) & np.isnan(prices)).any():
        raise SchemaError("units sold on a day without a price")

    weights = tables["weights"]
    if not np.isclose(weights.groupby("Level_id")["weight"].sum(), 1.0, atol=1e-4).all():
        raise SchemaError("official weights do not sum to one per level")

    day_names = [f"d_{d}" for d in range(1, n_days + 1)]
    sales.columns = day_names
    return {
        "series": series,
        "sales": sales,
        "calendar": calendar,
        "prices": pd.DataFrame(prices, columns=day_names),
        "weights": weights,
    }


def sample_series(processed: dict[str, pd.DataFrame], share: float, seed: int) -> dict[str, pd.DataFrame]:
    """Keep whole item and store series, stratified by store and department, with all their days.

    The upper levels of the hierarchy are aggregated later from the kept series. The official
    weights are kept as they are; the evaluation computes its own weights from the series.
    """
    series = processed["series"]
    keep = sample.select(
        series["item_id"] + "_" + series["store_id"],
        share,
        seed,
        strata=series["store_id"] + "_" + series["dept_id"],
    )
    out = dict(processed)
    for name in ("series", "sales", "prices"):
        out[name] = processed[name][keep].reset_index(drop=True)
    return out


def daily_prices(prices: pd.DataFrame, series: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """Weekly shelf prices as a series by day matrix; NaN before an item is first priced."""
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    week_index = {w: i for i, w in enumerate(weeks)}
    row = pd.MultiIndex.from_frame(series[["store_id", "item_id"]])
    rows = row.get_indexer(pd.MultiIndex.from_frame(prices[["store_id", "item_id"]]))
    if (rows < 0).any():
        raise SchemaError("prices list an item and store pair without sales")
    cols = prices["wm_yr_wk"].map(week_index)
    keep = cols.notna().to_numpy()
    weekly = np.full((len(series), len(weeks)), np.nan, dtype="float32")
    weekly[rows[keep], cols[keep].astype(int).to_numpy()] = prices["sell_price"].to_numpy()[keep]
    return weekly[:, calendar["wm_yr_wk"].map(week_index).to_numpy()]


def write(processed: dict[str, pd.DataFrame], data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "series": data_dir / SERIES_PATH.name,
        "sales": data_dir / SALES_PATH.name,
        "calendar": data_dir / CALENDAR_PATH.name,
        "prices": data_dir / PRICES_PATH.name,
        "weights": data_dir / OFFICIAL_WEIGHTS_PATH.name,
    }
    for key, path in paths.items():
        processed[key].to_parquet(path, index=False)


def load(data_dir: Path = DATA_DIR) -> dict:
    """Series ids, sales matrix (int32), prices matrix (float32) and calendar."""
    return {
        "series": pd.read_parquet(data_dir / SERIES_PATH.name),
        "sales": pd.read_parquet(data_dir / SALES_PATH.name).to_numpy(dtype="int32"),
        "prices": pd.read_parquet(data_dir / PRICES_PATH.name).to_numpy(dtype="float32"),
        "calendar": pd.read_parquet(data_dir / CALENDAR_PATH.name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="a local zip with the same eight files instead")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--expected-series", type=int, default=EXPECTED_SERIES, help="0 skips the size checks"
    )
    sample.add_arguments(parser, "item and store series, stratified by store and department")
    args = parse_args(parser, __file__)
    archive = args.source if args.source else download()
    processed = normalize(read_archive(archive), expected_series=args.expected_series or None)
    n_full = len(processed["series"])
    processed = sample_series(processed, args.sample, args.sample_seed)
    write(processed, args.data_dir)
    sample.record(args.data_dir, args.sample, args.sample_seed, "series", n_full, len(processed["series"]))

    from retail_forecast.demand import write_demand_tables

    write_demand_tables(args.data_dir)
    sales = processed["sales"]
    print(
        f"{len(sales):,} series x {sales.shape[1]:,} days written to {args.data_dir}; "
        f"{int(sales.to_numpy().sum()):,} units in total"
    )


if __name__ == "__main__":
    main()
