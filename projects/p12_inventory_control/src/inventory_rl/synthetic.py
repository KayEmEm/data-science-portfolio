"""A small synthetic archive in the M5 layout, for the tests and ``make smoke``.

Its numbers say nothing about Walmart. Two stores carry fast, medium and slow items with a
weekly pattern and negative binomial noise; the stores close on Christmas Day, as in M5.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from inventory_rl.config import EXPECTED_DAYS, SEED, SOURCE_FILES

STORES = {"CA_1": "CA", "TX_1": "TX"}
# department: (category, number of items, range of mean daily units, dispersion)
DEPARTMENTS = {
    "FOODS_3": ("FOODS", 6, (20.0, 80.0), 20.0),
    "FOODS_2": ("FOODS", 6, (3.0, 10.0), 3.0),
    "HOUSEHOLD_2": ("HOUSEHOLD", 6, (0.3, 0.9), 3.0),
}
WEEKLY = np.array([1.3, 1.25, 1.0, 0.9, 0.85, 0.85, 1.0])  # Saturday first, as wday 1 in M5


def make_calendar(n_days: int = EXPECTED_DAYS, start: str = "2011-01-29") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days)
    christmas = (dates.month == 12) & (dates.day == 25)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + np.arange(n_days) // 7,
            "weekday": dates.day_name(),
            "wday": (dates.day_of_week + 2) % 7 + 1,  # Saturday 1 ... Friday 7
            "month": dates.month,
            "year": dates.year,
            "event_name_1": np.where(christmas, "Christmas", None),
            "event_type_1": np.where(christmas, "National", None),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )


def make_tables(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calendar = make_calendar()
    wday = calendar["wday"].to_numpy()
    closed = calendar["event_name_1"].eq("Christmas").to_numpy()
    rows, units, prices = [], [], []
    for store, state in STORES.items():
        for dept, (category, count, (low, high), dispersion) in DEPARTMENTS.items():
            for i in range(1, count + 1):
                item = f"{dept}_{i:03d}"
                mean = rng.uniform(low, high) * WEEKLY[wday - 1]
                p = dispersion / (dispersion + mean)
                series = rng.negative_binomial(dispersion, p)
                series[closed] = 0
                rows.append(
                    {
                        "item_id": item,
                        "dept_id": dept,
                        "cat_id": category,
                        "store_id": store,
                        "state_id": state,
                    }
                )
                units.append(series)
                price = round(float(rng.uniform(1.0, 6.0)), 2)
                weeks = np.unique(calendar["wm_yr_wk"])
                prices.append(
                    pd.DataFrame({"store_id": store, "item_id": item, "wm_yr_wk": weeks, "sell_price": price})
                )
    ids = pd.DataFrame(rows)
    matrix = np.vstack(units)
    split = EXPECTED_DAYS - 28
    train = pd.concat(
        [ids, pd.DataFrame(matrix[:, :split], columns=[f"d_{d}" for d in range(1, split + 1)])], axis=1
    )
    test = pd.concat(
        [
            ids,
            pd.DataFrame(matrix[:, split:], columns=[f"d_{d}" for d in range(split + 1, EXPECTED_DAYS + 1)]),
        ],
        axis=1,
    )
    return {
        "calendar": calendar,
        "train": train,
        "test": test,
        "prices": pd.concat(prices, ignore_index=True),
    }


def write_archive(path: Path, seed: int = SEED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, table in make_tables(seed).items():
            buffer = io.StringIO()
            table.to_csv(buffer, index=False)
            archive.writestr(SOURCE_FILES[key], buffer.getvalue())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="zip file to write")
    args = parse_args(parser, __file__)
    print(write_archive(args.path))


if __name__ == "__main__":
    main()
