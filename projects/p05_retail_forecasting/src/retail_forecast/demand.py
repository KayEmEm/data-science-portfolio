"""Long daily demand tables per hierarchy level, for reuse outside this project.

Project 12 (inventory control) fits its demand distributions to these tables.
The interface is stable: ``daily_demand(level)`` returns one row per series and
day with the columns ``level``, ``series``, the level's id columns, ``d``,
``date`` and ``units``; level 12 also carries ``sell_price``. ``make data``
writes levels 1 to 9 to ``data/demand/level_<n>.parquet``; level 10 to 12 are
built on request from the wide sales table.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from retail_forecast.config import DATA_DIR, DEMAND_DIR, LEVELS, STORE_DEPT_LEVEL
from retail_forecast.data import load
from retail_forecast.hierarchy import build_level


def demand_frame(tables: dict, level: int) -> pd.DataFrame:
    """Daily units of every series of one level, in long format."""
    agg = build_level(tables["series"], level)
    units = agg.aggregate(tables["sales"])
    n_series, n_days = units.shape
    calendar = tables["calendar"]
    frame = pd.DataFrame(
        {
            "level": np.int8(level),
            "series": np.repeat(agg.names, n_days),
            "d": np.tile(calendar["d"].to_numpy(), n_series),
            "date": np.tile(calendar["date"].to_numpy(), n_series),
            "units": units.ravel().astype("int32"),
        }
    )
    for position, column in enumerate(LEVELS[level]):
        frame.insert(2 + position, column, np.repeat(agg.keys[column].to_numpy(), n_days))
    if level == 12:
        bottom = np.asarray(agg.matrix.argmax(axis=1)).ravel()
        frame["sell_price"] = tables["prices"][bottom].ravel()
    return frame


def write_demand_tables(data_dir: Path = DATA_DIR, levels=range(1, STORE_DEPT_LEVEL + 1)) -> list[Path]:
    tables = load(data_dir)
    out_dir = data_dir / DEMAND_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for level in levels:
        path = out_dir / f"level_{level}.parquet"
        demand_frame(tables, level).to_parquet(path, index=False)
        paths.append(path)
    return paths


def daily_demand(level: int = STORE_DEPT_LEVEL, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the long demand table of one level (1 to 12)."""
    if level not in LEVELS:
        raise ValueError(f"level must be 1 to 12, got {level}")
    path = data_dir / DEMAND_DIR.name / f"level_{level}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return demand_frame(load(data_dir), level)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, STORE_DEPT_LEVEL + 1)))
    args = parse_args(parser, __file__)
    for path in write_demand_tables(args.data_dir, args.levels):
        print(path)


if __name__ == "__main__":
    main()
