"""Download the UCI credit card default file, check its schema and store it as Parquet."""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from loan_default.config import (
    CLIENTS_PATH,
    ID_COL,
    RAW_DIR,
    REQUIRED_COLUMNS,
    SOURCE_TARGET,
    SOURCE_URL,
    TARGET_COL,
)

ARCHIVE_NAME = "default_of_credit_card_clients.zip"


class SchemaError(ValueError):
    """The source table lacks a required column."""


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch the zip archive from the UCI repository, unless it is already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME
    if not archive.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    return archive


def find_spreadsheet(raw_dir: Path) -> Path:
    """Extract every zip in ``raw_dir`` and return the one spreadsheet inside."""
    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
    sheets = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in {".xls", ".xlsx"})
    if len(sheets) != 1:
        raise FileNotFoundError(f"expected one spreadsheet under {raw_dir}, found {[p.name for p in sheets]}")
    return sheets[0]


def validate(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns {missing}")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and give the id and target their project names."""
    validate(frame)
    out = frame[REQUIRED_COLUMNS].rename(columns={"ID": ID_COL, SOURCE_TARGET: TARGET_COL})
    if out[ID_COL].duplicated().any():
        raise SchemaError("duplicate client ids")
    if not set(out[TARGET_COL].unique()) <= {0, 1}:
        raise SchemaError("target is not binary")
    return out.astype({c: "int64" for c in out.columns})


def read_spreadsheet(path: Path) -> pd.DataFrame:
    """Read the sheet, taking as header the first row that names ``LIMIT_BAL``.

    The header row is searched for rather than fixed, because the sheet may
    carry a row of variable codes (X1, X2, ...) above the column names.
    """
    preview = pd.read_excel(path, header=None, nrows=5, engine="calamine")
    matches = [i for i, row in preview.iterrows() if "LIMIT_BAL" in row.astype(str).tolist()]
    if not matches:
        raise SchemaError("no header row with LIMIT_BAL in the first five rows")
    return normalize(pd.read_excel(path, header=matches[0], engine="calamine"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="use the archive already in data/raw")
    args = parser.parse_args()
    if not args.skip_download:
        download()
    clients = read_spreadsheet(find_spreadsheet(RAW_DIR))
    CLIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clients.to_parquet(CLIENTS_PATH, index=False)
    print(
        f"{len(clients):,} clients written to {CLIENTS_PATH}; default rate {clients[TARGET_COL].mean():.4f}"
    )


if __name__ == "__main__":
    main()
