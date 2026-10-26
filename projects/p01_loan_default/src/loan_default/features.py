"""Build the feature table with DuckDB SQL over the client table."""

import argparse
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd
from portfolio_common.config import parse_args

from loan_default.config import CLIENTS_PATH, FEATURES_PATH, ID_COL


def read_sql(name: str) -> str:
    return resources.files("loan_default.sql").joinpath(f"{name}.sql").read_text()


def build_features(clients_path: Path = CLIENTS_PATH) -> pd.DataFrame:
    con = duckdb.connect()
    con.read_parquet(str(clients_path)).create_view("clients")
    frame = con.execute(read_sql("features")).df()
    if frame[ID_COL].duplicated().any():
        raise ValueError("feature table has duplicate client ids")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=Path, default=CLIENTS_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    args = parse_args(parser, __file__)
    frame = build_features(args.clients)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"{len(frame):,} clients and {frame.shape[1] - 2} features written to {args.out}")


if __name__ == "__main__":
    main()
