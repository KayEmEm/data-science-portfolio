"""Synthetic tables with the Home Credit schema, for tests and smoke runs only.

The values are random. Results computed on them say nothing about real
credit risk and must never be reported as project results.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loan_default.config import SEED


def make_tables(n_applicants: int = 2000, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n_applicants)
    ext = rng.uniform(0, 1, size=(n_applicants, 3))
    ext[rng.uniform(size=ext.shape) < 0.2] = np.nan
    income = rng.lognormal(11.8, 0.5, n_applicants)
    credit = income * rng.uniform(1, 6, n_applicants)
    days_birth = -rng.integers(21 * 365, 69 * 365, n_applicants)
    days_employed = np.where(
        rng.uniform(size=n_applicants) < 0.15, 365243, -rng.integers(0, 15000, n_applicants)
    )
    logit = -2.0 - 6.0 * (np.nan_to_num(ext, nan=0.5).mean(axis=1) - 0.5) + 0.3 * (credit / income - 3.5)
    target = (rng.uniform(size=n_applicants) < 1 / (1 + np.exp(-logit))).astype(int)
    application = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_applicants, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], n_applicants),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n_applicants),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit * rng.uniform(0.03, 0.08, n_applicants),
            "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n_applicants),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext[:, 0],
            "EXT_SOURCE_2": ext[:, 1],
            "EXT_SOURCE_3": ext[:, 2],
        }
    )

    n_bur = n_applicants * 3
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": rng.choice(ids, n_bur),
            "SK_ID_BUREAU": np.arange(5000000, 5000000 + n_bur),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], n_bur),
            "DAYS_CREDIT": -rng.integers(0, 3000, n_bur),
            "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 5, 30], n_bur),
            "AMT_CREDIT_SUM": rng.lognormal(12, 1, n_bur),
            "AMT_CREDIT_SUM_DEBT": rng.lognormal(11, 1, n_bur),
        }
    )
    n_bb = n_bur * 4
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": rng.choice(bureau["SK_ID_BUREAU"], n_bb),
            "MONTHS_BALANCE": -rng.integers(0, 60, n_bb),
            "STATUS": rng.choice(["0", "C", "X", "1", "2"], n_bb, p=[0.5, 0.3, 0.1, 0.07, 0.03]),
        }
    )
    n_prev = n_applicants * 2
    prev_ids = np.arange(2000000, 2000000 + n_prev)
    prev_curr = rng.choice(ids, n_prev)
    previous = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids,
            "SK_ID_CURR": prev_curr,
            "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], n_prev),
            "AMT_APPLICATION": rng.lognormal(11, 1, n_prev),
            "AMT_CREDIT": rng.lognormal(11, 1, n_prev),
            "DAYS_DECISION": -rng.integers(1, 2900, n_prev),
        }
    )

    def monthly(n_rows: int) -> pd.DataFrame:
        pick = rng.integers(0, n_prev, n_rows)
        return pd.DataFrame(
            {
                "SK_ID_PREV": prev_ids[pick],
                "SK_ID_CURR": prev_curr[pick],
                "MONTHS_BALANCE": -rng.integers(1, 96, n_rows),
            }
        )

    pos = monthly(n_prev * 5)
    pos["SK_DPD"] = rng.choice([0, 0, 0, 0, 3, 40], len(pos))
    pos["SK_DPD_DEF"] = np.where(pos["SK_DPD"] > 30, pos["SK_DPD"], 0)

    inst = monthly(n_prev * 6).drop(columns="MONTHS_BALANCE")
    inst["DAYS_INSTALMENT"] = -rng.integers(1, 2900, len(inst))
    inst["DAYS_ENTRY_PAYMENT"] = inst["DAYS_INSTALMENT"] + rng.integers(-10, 15, len(inst))
    inst["AMT_INSTALMENT"] = rng.lognormal(9, 0.8, len(inst))
    inst["AMT_PAYMENT"] = inst["AMT_INSTALMENT"] * rng.choice([1.0, 1.0, 1.0, 0.5], len(inst))

    cc = monthly(n_applicants * 3)
    cc["AMT_CREDIT_LIMIT_ACTUAL"] = rng.choice([0, 45000, 90000, 180000], len(cc))
    cc["AMT_BALANCE"] = cc["AMT_CREDIT_LIMIT_ACTUAL"] * rng.uniform(0, 1.1, len(cc))
    cc["SK_DPD"] = rng.choice([0, 0, 0, 7], len(cc))

    return {
        "application_train": application,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous,
        "POS_CASH_balance": pos,
        "installments_payments": inst,
        "credit_card_balance": cc,
    }


def write_tables(out_dir: Path, n_applicants: int = 2000, seed: int = SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in make_tables(n_applicants, seed).items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    write_tables(args.out_dir, args.n)


if __name__ == "__main__":
    main()
