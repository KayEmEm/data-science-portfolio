"""Build one feature row per applicant with DuckDB SQL over the seven source tables."""

import argparse
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

from loan_default.config import FEATURES_PATH, ID_COL, PARQUET_DIR, REQUIRED_COLUMNS
from loan_default.data import validate

# Aggregate queries joined to the application table, in join order.
AGGREGATE_QUERIES = ["bureau", "previous", "pos_cash", "installments", "credit_card"]


def read_sql(name: str) -> str:
    return resources.files("loan_default.sql").joinpath(f"{name}.sql").read_text()


def build_features(parquet_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """Register each Parquet table as a view, run the SQL files and join the results."""
    validate(parquet_dir, ".parquet")
    con = duckdb.connect()
    for table in REQUIRED_COLUMNS:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet_dir / f'{table}.parquet'}')"
        )
    con.execute(f"CREATE TEMP TABLE app AS {read_sql('application')}")
    joined = "SELECT * FROM app"
    for name in AGGREGATE_QUERIES:
        con.execute(f"CREATE TEMP TABLE agg_{name} AS {read_sql(name)}")
        joined += f" LEFT JOIN agg_{name} USING ({ID_COL})"
    frame = con.execute(f"{joined} ORDER BY {ID_COL}").df()
    if frame[ID_COL].duplicated().any():
        raise ValueError("feature table has duplicate applicant ids")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    frame = build_features(args.parquet_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"{len(frame):,} applicants and {frame.shape[1] - 2} features written to {args.out}")


if __name__ == "__main__":
    main()
