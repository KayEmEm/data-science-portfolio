"""Synthetic order lines in the Online Retail II layout, for the tests and ``make smoke``.

Customers are drawn from five types with different order rates, basket sizes and product mixes
(wholesale buyers, regular gift shops, Christmas buyers, single order customers, lapsed customers),
so that a clustering has structure to find. The file also holds what the cleaning must handle:
cancellations, lines without a customer id, postage lines, zero prices and exact duplicates.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2009-12-01")
END = pd.Timestamp("2011-12-09")
SOURCE_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

CATALOG = {
    "christmas": ["CHRISTMAS TREE DECORATION", "PAPER CHAIN KIT XMAS", "ADVENT CALENDAR", "SANTA BAUBLE"],
    "lights_candles": ["WHITE HANGING HEART T-LIGHT HOLDER", "GLASS LANTERN", "SCENTED CANDLE"],
    "bags": ["JUMBO BAG RED RETROSPOT", "LUNCH BAG SPACEBOY", "SHOPPER VINTAGE PAISLEY"],
    "kitchen_dining": ["REGENCY CAKESTAND 3 TIER", "ENAMEL MUG", "SET OF 4 PANTRY JELLY MOULDS"],
    "stationery_wrap": ["WRAP VINTAGE LEAF", "SET OF 3 NOTEBOOKS", "PACK OF 12 TISSUES"],
    "toys_crafts": ["SPACEBOY TOY", "JIGSAW BLOCKS", "FELTCRAFT DOLL"],
    "home_decor": ["PHOTO FRAME CORNICE", "WOODEN HEART SIGN", "DOOR MAT RED SPOT"],
    "garden_outdoor": ["GARDEN PARASOL", "BIRD HOUSE HANGING"],
    "other": ["ASSORTED COLOURS SILK FAN WIDGET", "SMALL POPCORN THING"],
}

TYPES = {
    # share, orders per year, lines per order, units per line, mix weights, active period
    "wholesale": (0.10, 30, 25, 24, {"kitchen_dining": 3, "bags": 3}, (0.0, 1.0)),
    "gift_shop": (0.30, 8, 12, 6, {"home_decor": 3, "lights_candles": 2, "stationery_wrap": 2}, (0.0, 1.0)),
    "christmas": (0.20, 3, 8, 6, {"christmas": 6, "lights_candles": 2}, (0.0, 1.0)),
    "single": (0.20, 1, 6, 3, {"toys_crafts": 2, "garden_outdoor": 2}, (0.0, 1.0)),
    "lapsed": (0.20, 6, 10, 6, {"kitchen_dining": 2, "home_decor": 2}, (0.0, 0.45)),
}


def product_table() -> pd.DataFrame:
    rows, code = [], 20000
    rng = np.random.default_rng(0)
    for category, names in CATALOG.items():
        for name in names:
            for variant in ("", " RED", " BLUE"):
                code += 7
                rows.append(
                    {
                        "code": str(code),
                        "description": name + variant,
                        "category": category,
                        "price": round(float(rng.uniform(0.4, 12.0)), 2),
                    }
                )
    return pd.DataFrame(rows)


def order_dates(rng, kind: str, n: int, period: tuple[float, float]) -> list[pd.Timestamp]:
    span = (END - START).total_seconds()
    if kind == "christmas":
        # Orders between September and early December of either year.
        out = []
        for _ in range(n):
            year = rng.choice([2010, 2011])
            day = pd.Timestamp(f"{year}-09-01") + pd.Timedelta(days=float(rng.uniform(0, 95)))
            out.append(min(day, END))
        return out
    lo, hi = period
    return [START + pd.Timedelta(seconds=float(rng.uniform(lo, hi)) * span) for _ in range(n)]


def make_lines(customers: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products = product_table()
    by_category = {c: g for c, g in products.groupby("category")}
    names = list(TYPES)
    shares = np.array([TYPES[t][0] for t in names])
    kinds = rng.choice(names, size=customers, p=shares / shares.sum())
    rows, invoice = [], 489000
    for i, kind in enumerate(kinds):
        _, per_year, lines_per_order, units, weights, period = TYPES[kind]
        customer = 12000 + i
        country = "United Kingdom" if rng.random() < 0.9 else rng.choice(["France", "Germany", "EIRE"])
        n_orders = 1 if kind == "single" else max(1, rng.poisson(per_year * 2))
        categories = list(by_category)
        w = np.array([1.0 + weights.get(c, 0) for c in categories])
        for date in order_dates(rng, kind, n_orders, period):
            invoice += 1
            n_lines = max(1, rng.poisson(lines_per_order))
            picks = rng.choice(categories, size=n_lines, p=w / w.sum())
            for category in picks:
                item = by_category[category].iloc[rng.integers(len(by_category[category]))]
                rows.append(
                    [
                        str(invoice),
                        item["code"],
                        item["description"],
                        max(1, rng.poisson(units)),
                        date,
                        item["price"],
                        float(customer),
                        country,
                    ]
                )
            if rng.random() < 0.05:
                rows.append([str(invoice), "POST", "POSTAGE", 1, date, 18.0, float(customer), country])
            if rng.random() < 0.08:
                cancel = rows[-1]
                rows.append(
                    [
                        f"C{invoice}",
                        cancel[1],
                        cancel[2],
                        -int(cancel[3]),
                        date + pd.Timedelta(days=2),
                        cancel[5],
                        float(customer),
                        country,
                    ]
                )
    frame = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    # Lines without a customer id, stock adjustments with a zero price, and exact duplicates.
    anonymous = frame.sample(frac=0.05, random_state=seed).assign(**{"Customer ID": np.nan})
    zero = frame.sample(n=10, random_state=seed + 1).assign(Price=0.0)
    duplicates = frame.sample(frac=0.01, random_state=seed + 2)
    frame = pd.concat([frame, anonymous, zero, duplicates], ignore_index=True)
    return frame.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)


def write_csv(folder: Path, frame: pd.DataFrame) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "online_retail_II.csv"
    frame.to_csv(path, index=False)
    return path


def write_xlsx(folder: Path, frame: pd.DataFrame) -> Path:
    """Two sheets split at December 2010, with the first days of December 2010 in both, as in the release."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "online_retail_II.xlsx"
    first = frame.loc[frame["InvoiceDate"] < pd.Timestamp("2010-12-10")]
    second = frame.loc[frame["InvoiceDate"] >= pd.Timestamp("2010-12-01")]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        first.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--customers", type=int, default=600)
    args = parser.parse_args()
    path = write_csv(args.folder, make_lines(args.customers))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
