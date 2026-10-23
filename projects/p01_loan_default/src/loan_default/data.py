"""Download the Home Credit tables, check their schema and store them as Parquet."""

import argparse
import subprocess
import zipfile
from pathlib import Path

import duckdb

from loan_default.config import KAGGLE_COMPETITION, PARQUET_DIR, RAW_DIR, REQUIRED_COLUMNS


class SchemaError(ValueError):
    """A source table is missing or lacks a required column."""


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download the competition archive with the Kaggle CLI and extract it.

    Requires Kaggle API credentials and acceptance of the competition rules
    on kaggle.com with the same account.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(raw_dir)],
        check=True,
    )
    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)


def validate(source_dir: Path, suffix: str = ".csv") -> None:
    """Raise ``SchemaError`` if a table or a required column is missing."""
    con = duckdb.connect()
    problems = []
    for table, required in REQUIRED_COLUMNS.items():
        path = source_dir / f"{table}{suffix}"
        if not path.exists():
            problems.append(f"{table}: file {path.name} not found")
            continue
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()}
        missing = sorted(set(required) - columns)
        if missing:
            problems.append(f"{table}: missing columns {missing}")
    if problems:
        raise SchemaError("; ".join(problems))


def to_parquet(raw_dir: Path = RAW_DIR, parquet_dir: Path = PARQUET_DIR) -> None:
    """Convert each required CSV table to Parquet for fast SQL access."""
    validate(raw_dir, ".csv")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for table in REQUIRED_COLUMNS:
        src = raw_dir / f"{table}.csv"
        dst = parquet_dir / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{src}', sample_size=-1)) TO '{dst}' (FORMAT PARQUET)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="use CSV files already in data/raw")
    args = parser.parse_args()
    if not args.skip_download:
        download()
    to_parquet()
    print(f"Parquet tables written to {PARQUET_DIR}")


if __name__ == "__main__":
    main()
