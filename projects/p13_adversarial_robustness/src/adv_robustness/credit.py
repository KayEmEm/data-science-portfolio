"""Feature manipulation against the credit default model of project 1, and two defenses.

The applicant sees only the decision (approve or reject) and may apply again with changed inputs,
so the attack is a black box search over the changes the applicant controls:

* ``payment``: an extra payment in the latest month, up to a share of the credit limit, borrowed
  elsewhere for the application (it raises PAY_AMT1 and lowers BILL_AMT1);
* ``self_reported``: any education and marital status the application form allows;
* ``both``: the two together.

The search keeps the cheapest change that turns a rejection into an approval. The model, its
calibration and its approval threshold come from project 1 (``fit_with_cv``, ``profit_curve``).
Defenses: the model without the self reported fields (they would be verified instead), adversarial
training on manipulated copies of the training defaulters, and both together.
"""

import argparse
import itertools
import json
from importlib import resources
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loan_default.config import EXPOSURE_COL, ID_COL, TARGET_COL, CostAssumptions, RunSettings
from loan_default.modeling import fit_with_cv
from loan_default.threshold import best_threshold, profit_curve
from portfolio_common.config import parse_args
from portfolio_common.evaluation import binary_metrics
from portfolio_common.seed import set_seed
from sklearn.model_selection import train_test_split

from adv_robustness.config import ARTIFACTS_DIR, DATA_DIR, SEED, CreditBudget

ATTACK_SETS = ("payment", "self_reported", "both")
EDUCATION_VALUES = (1, 2, 3, 4)  # graduate school, university, high school, others (UCI codebook)
MARRIAGE_VALUES = (1, 2, 3)  # married, single, others
VARIANTS = {
    "baseline": {"drop": (), "adversarial": False},
    "verified_fields": {"drop": ("EDUCATION", "MARRIAGE"), "adversarial": False},
    "adversarial_training": {"drop": (), "adversarial": True},
    "verified_and_adversarial": {"drop": ("EDUCATION", "MARRIAGE"), "adversarial": True},
}


def derive(clients: pd.DataFrame) -> pd.DataFrame:
    """The feature table of project 1 (its DuckDB SQL) for a client table in the source schema."""
    sql = resources.files("loan_default.sql").joinpath("features.sql").read_text()
    con = duckdb.connect()
    con.register("clients", clients)
    return con.execute(sql).df()


def candidates(clients: pd.DataFrame, attack: str, budget: CreditBudget) -> pd.DataFrame:
    """Every allowed change of every client, with its cost to the applicant in NT dollars.

    Each candidate keeps the client id in ``CLIENT_ID`` and its position in ``row``; the client id is
    then replaced by a unique key, because the feature SQL orders by it.
    """
    shares = (
        np.linspace(0.0, budget.extra_payment, budget.payment_steps) if attack != "self_reported" else [0.0]
    )
    fields = (
        list(itertools.product(EDUCATION_VALUES, MARRIAGE_VALUES)) if attack != "payment" else [(None, None)]
    )
    parts = []
    for share, (education, marriage) in itertools.product(shares, fields):
        part = clients.copy()
        extra = np.round(share * part[EXPOSURE_COL].to_numpy())
        part["PAY_AMT1"] = part["PAY_AMT1"] + extra
        part["BILL_AMT1"] = part["BILL_AMT1"] - extra
        if education is not None:
            part["EDUCATION"] = education
            part["MARRIAGE"] = marriage
        part["cost"] = extra
        part["row"] = np.arange(len(part))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["client"] = out[ID_COL]
    out[ID_COL] = np.arange(len(out))
    return out


def predict(model, clients: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of default for a client table in the source schema."""
    features = derive(clients.drop(columns=["cost", "row", "client"], errors="ignore"))
    features = features.set_index(ID_COL).loc[clients[ID_COL]].reset_index()
    return model.predict(features)


def attack(
    model, threshold: float, clients: pd.DataFrame, attack_set: str, budget: CreditBudget
) -> pd.DataFrame:
    """For each client, whether a change within the budget gets approval, and the cheapest such change."""
    pool = candidates(clients, attack_set, budget)
    pool["p"] = predict(model, pool)
    approved = pool[pool["p"] < threshold].sort_values(["row", "cost", "p"])
    best = approved.drop_duplicates("row").set_index("row")
    out = pd.DataFrame({"flipped": False, "cost": np.nan, "p_after": np.nan}, index=np.arange(len(clients)))
    out.loc[best.index, "flipped"] = True
    out.loc[best.index, "cost"] = best["cost"]
    out.loc[best.index, "p_after"] = best["p"]
    return out


def manipulated_copies(model, threshold: float, clients: pd.DataFrame, budget: CreditBudget) -> pd.DataFrame:
    """The cheapest approved change of each rejected client, under the strongest attack set."""
    rejected = clients[predict(model, clients.assign(row=0, cost=0.0, client=0)) >= threshold]
    rejected = rejected.reset_index(drop=True)
    pool = candidates(rejected, "both", budget)
    pool["p"] = predict(model, pool)
    best = pool[pool["p"] < threshold].sort_values(["row", "cost", "p"]).drop_duplicates("row")
    return best.drop(columns=["cost", "row", "p", "client"])


def fit_variant(
    train_clients: pd.DataFrame,
    settings: RunSettings,
    costs: CostAssumptions,
    drop: tuple,
    base=None,
    base_threshold: float | None = None,
    budget: CreditBudget | None = None,
):
    """A project 1 model on the training clients; with ``base``, also on manipulated defaulter copies."""
    rows = train_clients
    n_original = len(train_clients)
    if base is not None:
        defaulters = train_clients[train_clients[TARGET_COL] == 1]
        copies = manipulated_copies(base, base_threshold, defaulters, budget)
        copies[ID_COL] = np.arange(len(copies)) + train_clients[ID_COL].max() + 1
        rows = pd.concat([train_clients, copies], ignore_index=True)
    features = derive(rows).drop(columns=list(drop))
    order = features.set_index(ID_COL).loc[rows[ID_COL]].reset_index()
    fitted = fit_with_cv("lightgbm", order, settings)
    oof = fitted.calibrator.predict(fitted.oof_raw)[:n_original]
    y = train_clients[TARGET_COL].to_numpy()
    threshold = best_threshold(profit_curve(y, oof, train_clients[EXPOSURE_COL], costs))
    return fitted, threshold, len(rows) - n_original


class DroppingModel:
    """Removes the dropped columns before scoring, so the attack code sees one interface."""

    def __init__(self, fitted, drop: tuple):
        self.fitted, self.drop = fitted, list(drop)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.fitted.predict(features.drop(columns=self.drop))


def outcome(
    y: np.ndarray, p: np.ndarray, exposure: np.ndarray, threshold: float, costs: CostAssumptions
) -> dict:
    approved = p < threshold
    gain = np.where(y == 0, costs.margin * exposure, -costs.loss_given_default * exposure)
    return {
        "profit": float(gain[approved].sum()),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y[approved].mean()) if approved.any() else 0.0,
    }


def run(data_dir: Path, artifacts: Path, quick: bool = False, budget: CreditBudget | None = None) -> dict:
    set_seed(SEED)
    budget = budget or CreditBudget()
    settings = RunSettings.quick() if quick else RunSettings()
    costs = CostAssumptions()
    clients = pd.read_parquet(data_dir / "credit" / "clients.parquet")
    train, test = train_test_split(
        clients, test_size=settings.test_size, stratify=clients[TARGET_COL], random_state=SEED
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    y, exposure = test[TARGET_COL].to_numpy(), test[EXPOSURE_COL].to_numpy(dtype=float)
    summary = {"n_train": len(train), "n_test": len(test), "budget": budget.__dict__, "variants": {}}
    rows = []
    base = None
    for name, spec in VARIANTS.items():
        fitted, threshold, n_copies = fit_variant(
            train,
            settings,
            costs,
            spec["drop"],
            base=base["model"] if spec["adversarial"] else None,
            base_threshold=base["threshold"] if spec["adversarial"] else None,
            budget=budget,
        )
        model = DroppingModel(fitted, spec["drop"])
        p = predict(model, test.assign(row=0, cost=0.0, client=0))
        result = {
            "threshold": threshold,
            "added_training_copies": n_copies,
            "clean": binary_metrics(y, p) | outcome(y, p, exposure, threshold, costs),
            "attacks": {},
        }
        rejected = p >= threshold
        for attack_set in ATTACK_SETS:
            found = attack(model, threshold, test[rejected].reset_index(drop=True), attack_set, budget)
            flipped = found["flipped"].to_numpy()
            p_after = p.copy()
            p_after[np.flatnonzero(rejected)[flipped]] = found.loc[flipped, "p_after"].to_numpy()
            defaulters = y[rejected] == 1
            result["attacks"][attack_set] = {
                "rejected": int(rejected.sum()),
                "flipped": int(flipped.sum()),
                "flip_rate": float(flipped.mean()) if len(flipped) else 0.0,
                "flip_rate_defaulters": float(flipped[defaulters].mean()) if defaulters.any() else 0.0,
                "median_cost_share": float(
                    np.median(
                        found.loc[flipped, "cost"] / test.loc[rejected, EXPOSURE_COL].to_numpy()[flipped]
                    )
                )
                if flipped.any()
                else float("nan"),
                "after": outcome(y, p_after, exposure, threshold, costs),
            }
            rows.append({"variant": name, "attack": attack_set, **result["attacks"][attack_set]["after"]})
        summary["variants"][name] = result
        if name == "baseline":
            base = {"model": fitted, "threshold": threshold}
        print(
            f"credit {name}: AUC {result['clean']['roc_auc']:.3f}, flip rate (both) "
            f"{result['attacks']['both']['flip_rate']:.3f}"
        )
    out = artifacts / "credit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="small models, for tests and the smoke run")
    args = parse_args(parser, __file__)
    run(args.data_dir, args.artifacts, args.quick)


if __name__ == "__main__":
    main()
