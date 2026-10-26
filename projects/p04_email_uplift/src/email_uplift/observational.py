"""How assumptions change an estimate: the same question asked of observational data.

The randomized experiment gives the true average effect of the mens email.
This module builds an observational version of the same data: it keeps
customers as if a marketer had sent the email mostly to recent, high value
web customers, so that who got the email depends on who would visit anyway.
DoWhy then estimates the effect under three sets of assumptions:

* no adjustment (the naive comparison),
* adjustment for an incomplete set of confounders (past spend left out),
* adjustment for the full set of confounders the selection used.

The experiment is the benchmark each estimate is judged against. Refutation
tests check the main estimate.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dowhy import CausalModel
from portfolio_common.config import parse_args

from email_uplift.config import ARM_COL, ARMS, ARTIFACTS_DIR, CONTROL, CUSTOMERS_PATH, SEED
from email_uplift.data import feature_matrix, load
from email_uplift.experiment import difference_in_means

ARM = ARMS[0]  # mens email against no email
CONFOUNDERS = ["log_history", "recency", "channel_web", "newbie"]
INCOMPLETE = ["recency", "channel_web", "newbie"]  # past spend left out
METHODS = {
    "linear_regression": ("backdoor.linear_regression", {}),
    "propensity_weighting": (
        "backdoor.propensity_score_weighting",
        {"weighting_scheme": "ips_normalized_weight"},
    ),
    "propensity_stratification": ("backdoor.propensity_score_stratification", {"num_strata": 20}),
}
SELECTION_STRENGTH = {"log_history": 1.0, "recency": -0.6, "channel_web": 0.6, "newbie": -0.4}


def selection_probability(frame: pd.DataFrame) -> np.ndarray:
    """Probability that a customer would have been mailed under the simulated marketer rule."""
    features = feature_matrix(frame)
    score = sum(
        weight * (features[col] - features[col].mean()) / features[col].std()
        for col, weight in SELECTION_STRENGTH.items()
    )
    return 1 / (1 + np.exp(-score.to_numpy()))


def observational_sample(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Keep treated customers with probability p(x) and controls with probability 1 - p(x).

    Within the kept sample the chance of having been mailed rises with p(x),
    so the treated group is richer in customers who would visit anyway. The
    outcomes stay the real ones from the experiment.
    """
    pair = frame[frame[ARM_COL].isin([CONTROL, ARM])].reset_index(drop=True)
    p = selection_probability(pair)
    treated = (pair[ARM_COL] == ARM).to_numpy()
    keep = np.random.default_rng(seed).random(len(pair)) < np.where(treated, p, 1 - p)
    features = feature_matrix(pair)[CONFOUNDERS]
    out = pd.concat([features, pair[["visit", "spend"]]], axis=1)
    out["email"] = treated
    return out[keep].reset_index(drop=True)


def dowhy_estimate(sample: pd.DataFrame, outcome: str, confounders: list[str], method: str) -> dict:
    method_name, params = METHODS[method]
    # A copy per call: DoWhy writes a propensity_score column into the frame it is given
    # and reuses that column on later calls, whatever the confounders.
    model = CausalModel(data=sample.copy(), treatment="email", outcome=outcome, common_causes=confounders)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name=method_name, method_params=dict(params) or None)
    return {"effect": float(estimate.value), "model": model, "estimand": estimand, "estimate": estimate}


def refutations(sample: pd.DataFrame, outcome: str, confounders: list[str], n_simulations: int) -> list[dict]:
    """Placebo treatment, an added random confounder and a data subset, on one adjustment set.

    Run on the full and on the incomplete set, these show what the tests can
    and cannot detect: none of them reacts to a confounder left out.
    """
    result = dowhy_estimate(sample, outcome, confounders, "linear_regression")
    rows = []
    for method, params in (
        ("placebo_treatment_refuter", {"placebo_type": "permute"}),
        ("random_common_cause", {}),
        ("data_subset_refuter", {"subset_fraction": 0.8}),
    ):
        refuted = result["model"].refute_estimate(
            result["estimand"],
            result["estimate"],
            method_name=method,
            num_simulations=n_simulations,
            random_seed=SEED,
            **params,
        )
        p_value = refuted.refutation_result.get("p_value") if refuted.refutation_result else None
        rows.append(
            {
                "refuter": method,
                "estimated_effect": float(refuted.estimated_effect),
                "new_effect": float(refuted.new_effect),
                "p_value": None if p_value is None else float(p_value),
            }
        )
    return rows


def estimates_for_sample(sample: pd.DataFrame, outcome: str) -> dict[str, float]:
    """Naive difference and every DoWhy estimate on one observational sample."""
    treated = sample["email"]
    out = {
        "naive difference": float(sample.loc[treated, outcome].mean() - sample.loc[~treated, outcome].mean())
    }
    for method in METHODS:
        for label, confounders in (("incomplete", INCOMPLETE), ("full", CONFOUNDERS)):
            out[f"{method}, {label} adjustment"] = dowhy_estimate(sample, outcome, confounders, method)[
                "effect"
            ]
    return out


def run(
    customers_path: Path, artifacts_dir: Path, data_label: str, n_replicates: int, n_simulations: int
) -> dict:
    """Average every estimate over ``n_replicates`` independent selection draws.

    One draw keeps about half of the customers, so its estimates carry
    sampling noise as well as bias. The mean over draws isolates the bias;
    the spread shows the noise of a single study.
    """
    logging.getLogger("dowhy").setLevel(logging.WARNING)
    frame = load(customers_path)
    samples = [observational_sample(frame, SEED + r) for r in range(n_replicates)]
    p = selection_probability(frame[frame[ARM_COL].isin([CONTROL, ARM])])
    rows = []
    for outcome in ("visit", "spend"):
        truth = difference_in_means(
            frame.loc[frame[ARM_COL] == ARM, outcome], frame.loc[frame[ARM_COL] == CONTROL, outcome]
        )
        rows.append(
            {
                "outcome": outcome,
                "estimate": "experiment (benchmark)",
                "effect": truth["effect"],
                "sd": truth["se"],
            }
        )
        draws = pd.DataFrame([estimates_for_sample(sample, outcome) for sample in samples])
        for name in draws.columns:
            rows.append(
                {
                    "outcome": outcome,
                    "estimate": name,
                    "effect": float(draws[name].mean()),
                    "sd": float(draws[name].std(ddof=1)) if n_replicates > 1 else float("nan"),
                }
            )
        for row in rows:
            if row["outcome"] == outcome:
                row["bias"] = row["effect"] - truth["effect"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data": data_label,
        "arm": ARM,
        "n_replicates": n_replicates,
        "n_sample_mean": float(np.mean([len(s) for s in samples])),
        "share_mailed": float(np.mean([s["email"].mean() for s in samples])),
        "selection_probability_range": [float(p.min()), float(p.max())],
        "confounders": CONFOUNDERS,
        "incomplete_confounders": INCOMPLETE,
        "estimates": rows,
        "refutations_visit": [
            {"adjustment": label, **row}
            for label, confounders in (("full", CONFOUNDERS), ("incomplete", INCOMPLETE))
            for row in refutations(samples[0], "visit", confounders, n_simulations)
        ],
    }
    (artifacts_dir / "observational.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("--customers", type=Path, default=CUSTOMERS_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="hillstrom-2008", help="recorded with every output")
    parser.add_argument("--replicates", type=int, default=20, help="independent selection draws")
    parser.add_argument("--simulations", type=int, default=50, help="replicates per refutation test")
    args = parse_args(parser, __file__)
    summary = run(args.customers, args.artifacts, args.data_label, args.replicates, args.simulations)
    print(pd.DataFrame(summary["estimates"]).round(4).to_string(index=False))
    print(pd.DataFrame(summary["refutations_visit"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
