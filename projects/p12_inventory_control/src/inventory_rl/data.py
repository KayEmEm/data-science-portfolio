"""Download the M5 archive, select three products and fit their demand.

The archive is the one of project 5 (same URL and checksum). Three item x store series are
selected on the fit window by the rules in ``config.PROFILES``: a fast, a medium and a slow
(intermittent) mover. For each one, ``data/demand.parquet`` keeps the daily units and prices of
the fit window and the test year, and ``data/products.json`` keeps the fitted demand model
(``demand.py``), the costs and the capacity derived from it.

With ``--sample`` below 1 only the stores in ``config.SAMPLE_STORES`` are read and searched.
"""

import argparse
import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_rl import demand as demand_model
from inventory_rl.config import (
    CAPACITY_DAYS,
    DATA_DIR,
    DEMAND_PATH,
    EXPECTED_DAYS,
    FIT_DAYS,
    FIT_START,
    GROSS_MARGIN,
    HOLDING_RATE,
    ID_COLUMNS,
    MIN_CAPACITY,
    ORDER_COST,
    PRODUCTS_PATH,
    PROFILES,
    RAW_DIR,
    RAW_NAME,
    SAMPLE_STORES,
    SOURCE_FILES,
    SOURCE_SHA256,
    SOURCE_URL,
    TEST_DAYS,
    TEST_START,
)


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


def read_csv(archive: zipfile.ZipFile, key: str, stores: list[str] | None = None, **kwargs) -> pd.DataFrame:
    name = SOURCE_FILES[key]
    if name not in archive.namelist():
        raise SchemaError(f"{name} is missing from the archive")
    with archive.open(name) as handle:
        if stores is None:
            return pd.read_csv(handle, **kwargs)
        parts = [chunk[chunk["store_id"].isin(stores)] for chunk in pd.read_csv(handle, chunksize=2_000, **kwargs)]
    return pd.concat(parts, ignore_index=True)


def load_sales(archive: zipfile.ZipFile, stores: list[str] | None) -> tuple[pd.DataFrame, np.ndarray]:
    """The id columns and a (series x day) unit matrix for d_1 to d_1969."""
    train = read_csv(archive, "train", stores)
    test = read_csv(archive, "test", stores)
    for frame, first in ((train, 1), (test, None)):
        missing = [c for c in ID_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"sales table lacks {missing}")
        if first is not None and "d_1" not in frame.columns:
            raise SchemaError("sales table lacks d_1")
    merged = train.merge(test, on=ID_COLUMNS, how="inner", validate="one_to_one")
    days = [f"d_{i}" for i in range(1, EXPECTED_DAYS + 1)]
    absent = [d for d in days if d not in merged.columns]
    if absent:
        raise SchemaError(f"sales tables lack {len(absent)} day columns, first {absent[0]}")
    if len(merged) != len(train):
        raise SchemaError(f"{len(train) - len(merged)} series of the training file have no test window")
    units = merged[days].to_numpy(dtype="int64")
    if (units < 0).any():
        raise SchemaError("negative units in the sales table")
    return merged[ID_COLUMNS].reset_index(drop=True), units


def load_calendar(archive: zipfile.ZipFile) -> pd.DataFrame:
    calendar = read_csv(archive, "calendar")
    needed = {"date", "wm_yr_wk", "wday", "event_name_1"}
    if not needed <= set(calendar.columns):
        raise SchemaError(f"calendar lacks {sorted(needed - set(calendar.columns))}")
    if len(calendar) != EXPECTED_DAYS:
        raise SchemaError(f"calendar has {len(calendar)} days, expected {EXPECTED_DAYS}")
    calendar = calendar.copy()
    calendar["d"] = np.arange(1, EXPECTED_DAYS + 1)
    # All stores close on Christmas Day; those zeros are not demand.
    calendar["closed"] = calendar["event_name_1"].fillna("").eq("Christmas")
    return calendar


def candidate_table(ids: pd.DataFrame, units: np.ndarray) -> pd.DataFrame:
    fit = units[:, FIT_START - 1 : FIT_START - 1 + FIT_DAYS]
    sold = units > 0
    first_sale = np.where(sold.any(axis=1), sold.argmax(axis=1) + 1, EXPECTED_DAYS + 1)
    table = ids.copy()
    table["row"] = np.arange(len(ids))
    table["fit_mean"] = fit.mean(axis=1)
    table["zero_share"] = (fit == 0).mean(axis=1)
    table["first_sale"] = first_sale
    return table[table["first_sale"] <= FIT_START].reset_index(drop=True)


def ranked(candidates: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Candidates of one profile, closest to its target mean first."""
    low, high = profile["zero_share"]
    pool = candidates[candidates["zero_share"].between(low, high)]
    if pool.empty:
        return pool
    target = pool["fit_mean"].quantile(profile["mean_quantile"])
    order = np.lexsort((pool["row"].to_numpy(), (pool["fit_mean"] - target).abs().to_numpy()))
    return pool.iloc[order].assign(target_mean=target)


def daily_prices(archive: zipfile.ZipFile, keys: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Sell price per series and day for the given (store_id, item_id) pairs; NaN when absent."""
    wanted = set(zip(keys["store_id"], keys["item_id"], strict=True))
    prices = read_csv(archive, "prices", sorted({s for s, _ in wanted}))
    prices = prices[[pair in wanted for pair in zip(prices["store_id"], prices["item_id"], strict=True)]]
    return calendar[["d", "wm_yr_wk"]].merge(prices, on="wm_yr_wk", how="left")


def costs(mean_price: float, mean_units: float) -> dict:
    unit_cost = (1 - GROSS_MARGIN) * mean_price
    return {
        "mean_price": mean_price,
        "unit_cost": unit_cost,
        "holding": HOLDING_RATE * unit_cost / 365,  # per unit left on hand at the end of a day
        "lost_sale": mean_price - unit_cost,  # the margin lost on a unit not sold
        "order_cost": ORDER_COST,
        "capacity": max(MIN_CAPACITY, math.ceil(CAPACITY_DAYS * mean_units)),
    }


def select(archive: zipfile.ZipFile, stores: list[str] | None) -> tuple[pd.DataFrame, list[dict]]:
    ids, units = load_sales(archive, stores)
    calendar = load_calendar(archive)
    candidates = candidate_table(ids, units)
    window = calendar["d"].between(FIT_START, EXPECTED_DAYS)
    chosen, frames, used = [], [], set()
    for name, profile in PROFILES.items():
        pool = ranked(candidates, profile)
        pool = pool[~pool["row"].isin(used)]
        if pool.empty:
            raise SchemaError(f"no candidate series for the {name} profile")
        # Take the closest series whose price is known on every day from the fit start to the end.
        for start in range(0, len(pool), 20):
            block = pool.iloc[start : start + 20]
            prices = daily_prices(archive, block, calendar)
            found = None
            for _, row in block.iterrows():
                mine = prices[(prices["store_id"] == row["store_id"]) & (prices["item_id"] == row["item_id"])]
                mine = mine.set_index("d")["sell_price"].reindex(calendar["d"])
                if mine[window.to_numpy()].notna().all():
                    found = (row, mine.to_numpy())
                    break
            if found:
                break
        if not found:
            raise SchemaError(f"no {name} series has a price on every day of the fit and test windows")
        row, price = found
        used.add(int(row["row"]))
        series = units[int(row["row"])]
        part = calendar.loc[window, ["d", "date", "wday", "closed"]].copy()
        part.insert(0, "product", name)
        part["units"] = series[part["d"].to_numpy() - 1]
        part["sell_price"] = price[part["d"].to_numpy() - 1]
        frames.append(part)
        fit_part = part[part["d"] < TEST_START]
        model = demand_model.fit(fit_part["units"], fit_part["wday"], fit_part["closed"])
        mean_units = float(np.mean(model["weekday_mean"]))
        chosen.append(
            {
                "product": name,
                **{c: row[c] for c in ID_COLUMNS},
                "series_id": f"{row['item_id']}_{row['store_id']}",
                "selection": {
                    "fit_mean": float(row["fit_mean"]),
                    "zero_share": float(row["zero_share"]),
                    "target_mean": float(row["target_mean"]),
                    "candidates": len(pool),
                },
                "demand": model,
                "fit_check": demand_model.check(model, part, TEST_START),
                **costs(float(fit_part["sell_price"].mean()), mean_units),
            }
        )
    return pd.concat(frames, ignore_index=True), chosen


def write(source: Path, data_dir: Path, sample: float = 1.0) -> list[dict]:
    stores = SAMPLE_STORES if sample < 1 else None
    with zipfile.ZipFile(source) as archive:
        demand, products = select(archive, stores)
    data_dir.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(data_dir / DEMAND_PATH.name, index=False)
    meta = {"source": source.name, "stores": stores or "all", "test_days": TEST_DAYS, "products": products}
    (data_dir / PRODUCTS_PATH.name).write_text(json.dumps(meta, indent=2, default=str))
    return products


def load_products(data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / PRODUCTS_PATH.name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run make data first")
    return json.loads(path.read_text())["products"]


def load_demand(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / DEMAND_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="a local zip in the M5 layout instead of the download")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample", type=float, default=1.0, help="below 1: read only the sample stores")
    args = parser.parse_args()
    source = args.source or download(args.data_dir / RAW_DIR.name)
    for product in write(source, args.data_dir, args.sample):
        d = product["demand"]
        check = product["fit_check"]
        print(
            f"{product['product']}: {product['series_id']}, mean {np.mean(d['weekday_mean']):.2f} units per day, "
            f"dispersion {d['dispersion']:.2f}, price {product['mean_price']:.2f}, "
            f"test year mean ratio {check['test']['mean_ratio']:.2f}, "
            f"80% interval coverage {check['test']['coverage_80']:.1%}"
        )


if __name__ == "__main__":
    main()
