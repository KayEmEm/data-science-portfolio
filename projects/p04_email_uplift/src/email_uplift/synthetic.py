"""Synthetic customers in the Hillstrom layout, with known treatment effects.

Used by the tests and by ``make smoke``. The numbers say nothing about the
real campaign. The effects are built in so that tests can check recovery:
the mens email raises visits mainly for customers who bought mens
merchandise, the womens email mainly for womens buyers, and spend depends
on past spend so that CUPED has a covariate to work with.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from email_uplift.config import ARMS, CHANNELS, CONTROL, SEED, ZIP_CODES
from email_uplift.data import normalize

BASE_VISIT = 0.10
VISIT_LIFT = {"mens": 0.10, "womens": 0.08}  # for buyers of the matching category
OFF_TARGET_LIFT = 0.01
CONVERSION_GIVEN_VISIT = 0.08


def _history_segment(history: np.ndarray) -> np.ndarray:
    edges = [0, 100, 200, 350, 500, 750, 1000, np.inf]
    labels = [
        "1) $0 - $100",
        "2) $100 - $200",
        "3) $200 - $350",
        "4) $350 - $500",
        "5) $500 - $750",
        "6) $750 - $1,000",
        "7) $1,000 +",
    ]
    return np.asarray(labels, dtype=object)[np.digitize(history, edges[1:-1])]


def true_visit_effect(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """The visit probability lift built into the generator, per customer."""
    category = "mens" if arm == ARMS[0] else "womens"
    match = frame[category].to_numpy() == 1
    return np.where(match, VISIT_LIFT[category], OFF_TARGET_LIFT)


def make_customers(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = np.round(rng.lognormal(mean=5.0, sigma=0.9, size=n) + 29.99, 2)
    mens = rng.binomial(1, 0.55, n)
    womens = np.where(mens == 0, 1, rng.binomial(1, 0.45, n))  # every customer bought something
    frame = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": _history_segment(history),
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": rng.choice(ZIP_CODES, n, p=[0.4, 0.45, 0.15]),
            "newbie": rng.binomial(1, 0.5, n),
            "channel": rng.choice(CHANNELS, n, p=[0.44, 0.44, 0.12]),
            "segment": rng.choice([CONTROL, *ARMS], n),
        }
    )
    p_visit = BASE_VISIT + 0.04 * (frame["recency"].to_numpy() <= 3) + 0.03 * np.log(history / 100).clip(0)
    for arm in ARMS:
        treated = frame["segment"].to_numpy() == arm
        p_visit = p_visit + treated * true_visit_effect(frame, arm)
    visit = rng.binomial(1, p_visit.clip(0, 1))
    conversion = visit * rng.binomial(1, CONVERSION_GIVEN_VISIT, n)
    basket = 0.4 * history + rng.gamma(2.0, 40.0, n)
    frame["visit"] = visit
    frame["conversion"] = conversion
    frame["spend"] = np.round(conversion * basket, 2)
    return frame


def write_customers(path: Path, n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    customers = normalize(make_customers(n, seed), expected_rows=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    customers.to_parquet(path, index=False)
    return customers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=6000)
    args = parse_args(parser, __file__)
    customers = write_customers(args.out, args.n)
    print(f"{len(customers):,} synthetic customers written to {args.out}")


if __name__ == "__main__":
    main()
