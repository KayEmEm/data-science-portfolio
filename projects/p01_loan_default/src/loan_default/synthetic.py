"""A synthetic client table with the UCI source schema, for tests and smoke runs only.

The values are random. Results computed on them say nothing about real
credit risk and must never be reported as project results.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loan_default.config import MONTHS, SEED, SOURCE_TARGET


def make_source(n_clients: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Columns and value ranges follow the UCI spreadsheet, before renaming."""
    rng = np.random.default_rng(seed)
    limit = rng.choice(np.arange(10_000, 800_001, 10_000), n_clients)
    risk = rng.normal(0, 1, n_clients)
    delays = np.clip(np.round(risk[:, None] + rng.normal(0, 1, (n_clients, 6))), -2, 8).astype(int)
    util = np.clip(0.4 + 0.2 * risk[:, None] + rng.normal(0, 0.2, (n_clients, 6)), -0.1, 1.2)
    bills = np.round(util * limit[:, None]).astype(int)
    paid_share = np.clip(rng.uniform(0, 0.5, (n_clients, 6)) - 0.1 * risk[:, None], 0, 1)
    payments = np.round(np.maximum(bills, 0) * paid_share).astype(int)
    logit = -1.6 + 1.1 * risk + 0.3 * (delays[:, 0] >= 1)
    target = (rng.uniform(size=n_clients) < 1 / (1 + np.exp(-logit))).astype(int)

    frame = pd.DataFrame(
        {
            "ID": np.arange(1, n_clients + 1),
            "LIMIT_BAL": limit,
            "SEX": rng.choice([1, 2], n_clients, p=[0.4, 0.6]),
            "EDUCATION": rng.choice([1, 2, 3, 4], n_clients, p=[0.35, 0.47, 0.16, 0.02]),
            "MARRIAGE": rng.choice([1, 2, 3], n_clients, p=[0.45, 0.53, 0.02]),
            "AGE": rng.integers(21, 76, n_clients),
        }
    )
    for i, name in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        frame[name] = delays[:, i]
    for m in MONTHS:
        frame[f"BILL_AMT{m}"] = bills[:, m - 1]
    for m in MONTHS:
        frame[f"PAY_AMT{m}"] = payments[:, m - 1]
    frame[SOURCE_TARGET] = target
    return frame


def write_clients(out: Path, n_clients: int = 3000, seed: int = SEED) -> None:
    from loan_default.data import normalize

    out.parent.mkdir(parents=True, exist_ok=True)
    normalize(make_source(n_clients, seed)).to_parquet(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output Parquet file")
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()
    write_clients(args.out, args.n)


if __name__ == "__main__":
    main()
