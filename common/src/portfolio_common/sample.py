"""Seeded samples of whole units (rows, users, series, time steps) for smaller runs.

A unit is kept when its hash rank inside its stratum falls in the first ``fraction`` of the
stratum. The rank depends only on the unit key and the seed, so the same call gives the same
sample on every machine, and the sample at a smaller fraction is part of the sample at a larger
one. Every stratum keeps at least one unit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FULL = 1.0
DEFAULT_SEED = 42
RECORD_NAME = "sample.json"


def fraction(value: str) -> float:
    """Argument type: a share of the data above 0 and at most 1."""
    share = float(value)
    if not 0.0 < share <= 1.0:
        raise argparse.ArgumentTypeError(f"sample must be above 0 and at most 1, got {value}")
    return share


def add_arguments(parser: argparse.ArgumentParser, unit: str) -> None:
    """Add ``--sample`` and ``--sample-seed`` to a data step."""
    parser.add_argument(
        "--sample",
        type=fraction,
        default=FULL,
        help=f"share of {unit} to keep, seeded; 1 (the default) keeps the full data",
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SEED)


def unit_hash(keys: pd.Series, seed: int = DEFAULT_SEED) -> np.ndarray:
    """A stable 64 bit hash per key; it depends only on the key text and the seed."""
    hash_key = f"{seed:016d}"[-16:]
    return pd.util.hash_pandas_object(keys.astype(str), index=False, hash_key=hash_key).to_numpy()


def select(
    keys: pd.Series,
    share: float,
    seed: int = DEFAULT_SEED,
    strata: pd.Series | None = None,
    always: list | None = None,
) -> np.ndarray:
    """Boolean mask over ``keys``: True for every row of a kept unit.

    ``strata`` gives the stratum of each row (one stratum per unit); ``always`` lists units kept
    whatever their rank.
    """
    if share >= FULL:
        return np.ones(len(keys), dtype=bool)
    units = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "stratum": strata.to_numpy() if strata is not None else 0,
        }
    ).drop_duplicates("key")
    units["hash"] = unit_hash(units["key"], seed)
    units["rank"] = units.groupby("stratum", sort=False)["hash"].rank(method="first")
    size = units.groupby("stratum", sort=False)["key"].transform("size")
    keep = units["rank"] <= np.maximum(1, np.ceil(size * share))
    kept = set(units.loc[keep, "key"]) | set(always or [])
    return keys.isin(kept).to_numpy()


def record(directory: Path, share: float, seed: int, unit: str, before: int, after: int) -> dict:
    """Write ``sample.json`` beside the processed data and return its content."""
    content = {
        "fraction": share,
        "seed": seed,
        "unit": unit,
        "units_before": int(before),
        "units_after": int(after),
        "full_data": share >= FULL,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RECORD_NAME).write_text(json.dumps(content, indent=2))
    return content
