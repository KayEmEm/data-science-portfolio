"""A small synthetic archive in the M5 layout, for the tests and ``make smoke``.

Its numbers say nothing about Walmart. Units are Poisson with a weekly
pattern, a price effect and SNAP days; some items start selling late, and the
stores close on Christmas Day, as in M5.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from retail_forecast.config import SEED, SOURCE_FILES

STORES = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX"}
DEPARTMENTS = {"FOODS_1": "FOODS", "FOODS_2": "FOODS", "HOBBIES_1": "HOBBIES"}
WEEKLY = np.array([1.35, 1.3, 1.0, 0.9, 0.85, 0.9, 1.05])  # Saturday first, as wday 1 in M5


def make_calendar(n_days: int, start: str = "2011-01-29") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days)
    rng = np.random.default_rng(SEED)
    weekday = dates.day_name()
    wday = (dates.day_of_week + 2) % 7 + 1  # Saturday 1 ... Friday 7
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + (np.arange(n_days) // 7),
            "weekday": weekday,
            "wday": wday,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": np.where(
                (dates.month == 12) & (dates.day == 25),
                "Christmas",
                np.where((dates.month == 2) & (dates.day == 14), "ValentinesDay", None),
            ),
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    calendar["event_type_1"] = calendar["event_name_1"].map(
        {"Christmas": "National", "ValentinesDay": "Cultural"}
    )
    for state in ("CA", "TX", "WI"):
        calendar[f"snap_{state}"] = ((dates.day <= 10) & (rng.random(n_days) < 0.95)).astype(int)
    columns = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "event_name_1", "event_type_1"]
    return calendar[[*columns, "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]]


def make_tables(n_days: int = 500, items_per_dept: int = 6, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calendar = make_calendar(n_days)
    rows = []
    for dept, cat in DEPARTMENTS.items():
        for k in range(1, items_per_dept + 1):
            for store, state in STORES.items():
                rows.append(
                    {
                        "item_id": f"{dept}_{k:03d}",
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": state,
                    }
                )
    series = pd.DataFrame(rows)
    n = len(series)
    level = rng.gamma(1.5, 2.0, n) * np.where(series["cat_id"] == "FOODS", 2.0, 0.7)
    release = np.where(rng.random(n) < 0.2, rng.integers(30, n_days // 3, n), 0)
    weeks = calendar["wm_yr_wk"].to_numpy()
    week_ids = np.unique(weeks)
    base_price = rng.uniform(1.0, 12.0, n)
    weekly_price = base_price[:, None] * np.where(rng.random((n, len(week_ids))) < 0.1, 0.8, 1.0)
    price = weekly_price[:, np.searchsorted(week_ids, weeks)]
    wday = calendar["wday"].to_numpy() - 1
    trend = 1 + 0.3 * np.arange(n_days) / n_days
    states = series["state_id"].to_numpy()
    snap = np.stack([calendar[f"snap_{s}"].to_numpy() for s in states])
    rate = level[:, None] * WEEKLY[wday][None, :] * trend[None, :] * (price / base_price[:, None]) ** -1.5
    rate *= 1 + 0.3 * snap
    days = np.arange(n_days)[None, :]
    rate = np.where(days >= release[:, None], rate, 0.0)
    christmas = (calendar["event_name_1"] == "Christmas").to_numpy()
    rate[:, christmas] = 0.0
    units = rng.poisson(rate)

    day_names = [f"d_{d}" for d in range(1, n_days + 1)]
    sales = pd.concat([series, pd.DataFrame(units, columns=day_names)], axis=1)
    split = n_days - 28
    train = sales[[*series.columns, *day_names[:split]]]
    test = sales[[*series.columns, *day_names[split:]]]

    priced = []
    for i, row in series.iterrows():
        first_week = weeks[release[i]]
        keep = week_ids >= first_week
        priced.append(
            pd.DataFrame(
                {
                    "store_id": row["store_id"],
                    "item_id": row["item_id"],
                    "wm_yr_wk": week_ids[keep],
                    "sell_price": np.round(weekly_price[i, keep], 2),
                }
            )
        )
    prices = pd.concat(priced, ignore_index=True)
    dollars = (units[:, split - 28 : split] * price[:, split - 28 : split]).sum(axis=1)
    weights = pd.DataFrame(
        {
            "Level_id": "Level12",
            "Agg_Level_1": series["item_id"],
            "Agg_Level_2": series["store_id"],
            "Dollar_Sales": dollars,
            "weight": dollars / dollars.sum(),
        }
    )
    return {"calendar": calendar, "prices": prices, "train": train, "test": test, "weights": weights}


def write_archive(path: Path, n_days: int = 500, items_per_dept: int = 6) -> Path:
    tables = make_tables(n_days, items_per_dept)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, name in SOURCE_FILES.items():
            buffer = io.StringIO()
            tables[key].to_csv(buffer, index=False)
            archive.writestr(name, buffer.getvalue())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--items-per-dept", type=int, default=6)
    args = parse_args(parser, __file__)
    print(write_archive(args.out, args.days, args.items_per_dept))


if __name__ == "__main__":
    main()
