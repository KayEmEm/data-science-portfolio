"""Read the Online Retail II transactions, check and clean them, and store one table of order lines.

The UCI release is ``online_retail_II.xlsx`` with one sheet per year (2009 to 2010 and 2010 to 2011);
a CSV export of both sheets is read as well, and ``--source`` may name a zip that holds either, in
any subfolder. Cleaning, each step counted in ``data/audit.json``:

* exact duplicate rows are dropped;
* rows without a customer id are dropped (they cannot be assigned to a customer);
* invoices that start with "C" are cancellations: their lines are kept as returns with negative
  quantities; other lines with a quantity or price of 0 or less are dropped (stock adjustments);
* lines whose stock code is not a product code (postage, fees, discounts, manual adjustments) are
  flagged and kept out of the spend features;
* each product line gets a category from its description (``categories.py``).
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample
from portfolio_common.config import parse_args

from customer_segments.categories import OTHER, categorize_series
from customer_segments.config import (
    AUDIT_PATH,
    COLUMNS,
    DATA_DIR,
    DOCUMENTED_ROWS,
    PRODUCT_CODE,
    RAW_DIR,
    RENAME,
    SOURCE_NAMES,
    TRANSACTIONS_PATH,
)


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def normalize_name(name: str) -> str:
    return "".join(str(name).lower().split()).replace("_", "")


def find_source(source: Path) -> Path:
    if source.is_file():
        return source
    for name in SOURCE_NAMES:
        if (source / name).exists():
            return source / name
    found = sorted(p for p in source.glob("*") if p.suffix.lower() in {".xlsx", ".csv", ".zip"})
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"{source} holds none of {SOURCE_NAMES}; download the release from the UCI page (see the README)"
    )


def extract(archive: Path, target: Path) -> Path:
    """Extract the workbook or CSV file from a zip, wherever it sits in it, into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = [
            m
            for m in handle.infolist()
            if not m.is_dir()
            and Path(m.filename).suffix.lower() in {".xlsx", ".csv"}
            and not Path(m.filename).name.startswith(".")
        ]
        if not members:
            raise FileNotFoundError(f"{archive} holds no .xlsx or .csv file")
        member = max(members, key=lambda m: m.file_size)
        out = target / Path(member.filename).name
        with handle.open(member) as source, out.open("wb") as sink:
            while block := source.read(1 << 20):
                sink.write(block)
    return out


def read_raw(path: Path) -> pd.DataFrame:
    """All rows of every sheet (or of the CSV file), with a column naming the sheet."""
    text = {"Invoice": str, "InvoiceNo": str, "StockCode": str, "Description": str, "Country": str}
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=text, encoding="utf-8-sig", encoding_errors="replace")
        frame["sheet"] = path.name
        return frame
    sheets = pd.read_excel(path, sheet_name=None, dtype=text, engine="openpyxl")
    return pd.concat([part.assign(sheet=name) for name, part in sheets.items()], ignore_index=True)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: RENAME[normalize_name(c)] for c in raw.columns if normalize_name(c) in RENAME}
    frame = raw.rename(columns=renamed)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"the source lacks the columns {missing}")
    out = pd.DataFrame(
        {
            "invoice": frame["invoice"].astype(str).str.strip(),
            "stock_code": frame["stock_code"].astype(str).str.strip(),
            "description": frame["description"].fillna("").astype(str).str.strip(),
            "quantity": pd.to_numeric(frame["quantity"], errors="coerce"),
            "invoice_date": pd.to_datetime(frame["invoice_date"], errors="coerce", dayfirst=False),
            "price": pd.to_numeric(frame["price"], errors="coerce"),
            "customer_id": pd.to_numeric(frame["customer_id"], errors="coerce"),
            "country": frame["country"].fillna("Unspecified").astype(str).str.strip(),
            "sheet": frame["sheet"].astype(str),
        }
    )
    bad = out[["quantity", "invoice_date", "price"]].isna().any(axis=1)
    if bad.mean() > 0.01:
        raise SchemaError(f"{bad.sum():,} rows have a quantity, date or price that does not parse")
    return out.loc[~bad].reset_index(drop=True)


def clean(lines: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit: dict = {"rows_read": len(lines)}
    value = lines["quantity"] * lines["price"]
    audit["value_read"] = round(float(value.sum()), 2)

    content = [c for c in lines.columns if c != "sheet"]
    duplicate = lines.duplicated(subset=content)
    audit["duplicate_rows"] = int(duplicate.sum())
    if lines["sheet"].nunique() > 1:
        # Rows that appear in two sheets, a count of the overlap between the yearly sheets.
        first = lines.drop_duplicates(subset=content + ["sheet"])
        audit["rows_in_two_sheets"] = int(first.duplicated(subset=content).sum())
    lines = lines.loc[~duplicate]

    no_customer = lines["customer_id"].isna()
    audit["rows_without_customer"] = int(no_customer.sum())
    audit["value_without_customer"] = round(float((lines["quantity"] * lines["price"])[no_customer].sum()), 2)
    lines = lines.loc[~no_customer].copy()

    lines["is_return"] = lines["invoice"].str.upper().str.startswith("C")
    adjustment = ~lines["is_return"] & ((lines["quantity"] <= 0) | (lines["price"] <= 0))
    audit["non_positive_lines_dropped"] = int(adjustment.sum())
    lines = lines.loc[~adjustment].copy()
    # A return line with a positive quantity would add spend; it is turned into a negative one.
    lines.loc[lines["is_return"], "quantity"] = -lines.loc[lines["is_return"], "quantity"].abs()
    lines = lines.loc[lines["price"] > 0].copy()

    lines["is_product"] = lines["stock_code"].str.match(PRODUCT_CODE)
    lines["value"] = lines["quantity"] * lines["price"]
    lines["customer_id"] = lines["customer_id"].astype("int64")
    lines["category"] = np.where(lines["is_product"], categorize_series(lines["description"]), "non_product")

    products = lines.loc[lines["is_product"] & ~lines["is_return"]]
    non_product = lines.loc[~lines["is_product"]]
    spend_by_category = products.groupby("category")["value"].sum()
    audit.update(
        {
            "rows_kept": len(lines),
            "return_lines": int(lines["is_return"].sum()),
            "non_product_lines": len(non_product),
            "non_product_value": round(float(non_product["value"].sum()), 2),
            "non_product_codes": non_product["stock_code"].str.upper().value_counts().head(15).to_dict(),
            "customers": int(lines["customer_id"].nunique()),
            "invoices": int(lines["invoice"].nunique()),
            "countries": int(lines["country"].nunique()),
            "first_date": str(lines["invoice_date"].min()),
            "last_date": str(lines["invoice_date"].max()),
            "category_spend_share": (spend_by_category / spend_by_category.sum()).round(4).to_dict(),
            "other_category_spend_share": round(
                float(spend_by_category.get(OTHER, 0.0) / spend_by_category.sum()), 4
            ),
        }
    )
    return lines.drop(columns=["sheet"]).reset_index(drop=True), audit


def prepare(source: Path) -> tuple[pd.DataFrame, dict]:
    path = find_source(source)
    if path.suffix.lower() == ".zip":
        path = extract(path, RAW_DIR)
    raw = read_raw(path)
    lines, audit = clean(normalize(raw))
    audit = {
        "source_file": path.name,
        "sheets": sorted(raw["sheet"].unique().tolist()),
        "documented_rows": DOCUMENTED_ROWS,
        "rows_match_documented": len(raw) == DOCUMENTED_ROWS,
        **audit,
    }
    return lines, audit


def write(lines: pd.DataFrame, audit: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(data_dir / TRANSACTIONS_PATH.name, index=False)
    (data_dir / AUDIT_PATH.name).write_text(json.dumps(audit, indent=2, default=str))


def load(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / TRANSACTIONS_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=RAW_DIR, help="the workbook, a CSV export, a zip, or their folder"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    sample.add_arguments(parser, "customers, each with all their order lines, stratified by country")
    args = parse_args(parser, __file__)
    lines, audit = prepare(args.source)
    # The audit describes the whole source; the sample is drawn after it.
    n_full = lines["customer_id"].nunique()
    keep = sample.select(lines["customer_id"], args.sample, args.sample_seed, strata=lines["country"])
    lines = lines[keep].reset_index(drop=True)
    write(lines, audit, args.data_dir)
    sample.record(
        args.data_dir, args.sample, args.sample_seed, "customer", n_full, lines["customer_id"].nunique()
    )
    print(
        f"{audit['rows_read']:,} rows read (documented {DOCUMENTED_ROWS:,}), {audit['rows_kept']:,} kept; "
        f"{audit['customers']:,} customers, {audit['invoices']:,} invoices, "
        f"{audit['first_date'][:10]} to {audit['last_date'][:10]}; "
        f"spend without a category rule: {audit['other_category_spend_share']:.1%}"
    )


if __name__ == "__main__":
    main()
