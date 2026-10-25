"""Download the Hillstrom email experiment, check its schema and store it as Parquet."""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample

from email_uplift.config import (
    ARM_COL,
    ARMS,
    CHANNELS,
    CONTROL,
    CUSTOMERS_PATH,
    EXPECTED_ROWS,
    OUTCOMES,
    RAW_DIR,
    RAW_NAME,
    REQUIRED_COLUMNS,
    SOURCE_SHA256,
    SOURCE_URL,
    ZIP_CODES,
)

TREATMENT_COL = "treatment"  # 0 control, 1 mens email, 2 womens email
TREATMENT_CODES = {CONTROL: 0, **{arm: i + 1 for i, arm in enumerate(ARMS)}}


class SchemaError(ValueError):
    """The source table does not match the documented layout."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch the gzip CSV, unless present, and check that it is the pinned file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_NAME
    if not target.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    found = sha256(target)
    if found != SOURCE_SHA256:
        raise SchemaError(f"{target} has SHA256 {found}, expected {SOURCE_SHA256}")
    return target


def normalize(frame: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Check columns, categories and value ranges; add the integer treatment code."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns {missing}")
    out = frame[REQUIRED_COLUMNS].copy()
    if expected_rows is not None and len(out) != expected_rows:
        raise SchemaError(f"expected {expected_rows} rows, found {len(out)}")
    if out.isna().any().any():
        raise SchemaError(f"missing values in {out.columns[out.isna().any()].tolist()}")
    for col, allowed in ((ARM_COL, [CONTROL, *ARMS]), ("zip_code", ZIP_CODES), ("channel", CHANNELS)):
        unknown = set(out[col].unique()) - set(allowed)
        if unknown:
            raise SchemaError(f"unknown values in {col}: {sorted(unknown)}")
    for col in ("mens", "womens", "newbie", "visit", "conversion"):
        if not set(out[col].unique()) <= {0, 1}:
            raise SchemaError(f"{col} is not binary")
    if not out["recency"].between(1, 12).all():
        raise SchemaError("recency outside 1 to 12 months")
    if (out["history"] < 0).any() or (out["spend"] < 0).any():
        raise SchemaError("negative spend")
    if (out.loc[out["conversion"] == 0, "spend"] > 0).any():
        raise SchemaError("spend recorded without a conversion")
    out[TREATMENT_COL] = out[ARM_COL].map(TREATMENT_CODES).astype("int64")
    ints = ["recency", "mens", "womens", "newbie", *OUTCOMES[:2]]
    return out.astype({c: "int64" for c in ints} | {"history": "float64", "spend": "float64"})


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric pre treatment features: the raw columns plus one hot zip code and channel."""
    out = frame[["recency", "history", "mens", "womens", "newbie"]].astype("float64")
    out["log_history"] = np.log1p(frame["history"])
    for value in ZIP_CODES[1:]:
        out[f"zip_{value.lower()}"] = (frame["zip_code"] == value).astype("float64")
    for value in CHANNELS[1:]:
        out[f"channel_{value.lower()}"] = (frame["channel"] == value).astype("float64")
    return out


def load(path: Path = CUSTOMERS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="a local copy of the CSV (plain or gzip) instead")
    parser.add_argument("--out", type=Path, default=CUSTOMERS_PATH)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS, help="0 skips the row check")
    sample.add_arguments(parser, "customers (rows of the source file), stratified by arm")
    args = parser.parse_args()
    source = args.source if args.source else download()
    customers = normalize(pd.read_csv(source), expected_rows=args.expected_rows or None)
    n_full = len(customers)
    row = pd.Series(np.arange(n_full))
    keep = sample.select(row, args.sample, args.sample_seed, strata=customers[ARM_COL])
    customers = customers[keep].reset_index(drop=True)
    sample.record(args.out.parent, args.sample, args.sample_seed, "customer", n_full, len(customers))
    customers.to_parquet(args.out, index=False)
    shares = customers[ARM_COL].value_counts().to_dict()
    print(f"{len(customers):,} customers written to {args.out}; arms {shares}")


if __name__ == "__main__":
    main()
