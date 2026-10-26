"""Score the uplift models and the targeting policies once on the held out test split.

Qini coefficients and policy values come with bootstrap intervals, because
on 19,200 customers the differences between models are small next to the
noise in two weeks of spend.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from email_uplift.config import ARTIFACTS_DIR, SEED, Economics
from email_uplift.data import TREATMENT_COL
from email_uplift.targeting import (
    bootstrap_policy_values,
    policy_value,
    qini_coefficient,
    qini_curve,
    rewards,
)
from email_uplift.train import ARM_NAMES, OUTCOMES, POLICY_OUTCOME, candidate_policies
from email_uplift.uplift import ARM_CODES, BASELINES

COST_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def score_columns(test: pd.DataFrame, name: str, outcome: str) -> list[str]:
    columns = [f"score_{name}_{arm}" for arm in ARM_NAMES]
    if columns[0] in test.columns:  # baselines do not depend on the outcome
        return columns
    return [f"score_{outcome}_{name}_{arm}" for arm in ARM_NAMES]


def qini_with_interval(
    score: np.ndarray, y: np.ndarray, treated: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> dict:
    point = qini_coefficient(score, y, treated)
    idx_t, idx_c = np.flatnonzero(treated), np.flatnonzero(~treated)
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        rows = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        draws[b] = qini_coefficient(score[rows], y[rows], treated[rows])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"qini": point, "ci_low": float(low), "ci_high": float(high)}


def random_band(y: np.ndarray, treated: np.ndarray, n_orderings: int, rng: np.random.Generator) -> dict:
    """Qini coefficients of random orderings: their mean is zero, their spread is the noise floor.

    A model whose coefficient sits inside this band ranks customers no
    better than chance could on a test split of this size.
    """
    draws = np.array([qini_coefficient(rng.random(len(y)), y, treated) for _ in range(n_orderings)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"qini": float(draws.mean()), "ci_low": float(low), "ci_high": float(high)}


def qini_results(test: pd.DataFrame, names: list[str], n_bootstrap: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SEED)
    t = test[TREATMENT_COL].to_numpy()
    rows, curves = [], {}
    for outcome in OUTCOMES:
        y = test[outcome].to_numpy(dtype=float)
        for name in names:
            columns = score_columns(test, name, outcome)
            for j, (arm, code) in enumerate(zip(ARM_NAMES, ARM_CODES, strict=True)):
                pair = np.isin(t, [0, code])
                score = test[columns[j]].to_numpy()[pair]
                treated = t[pair] == code
                if name == "random":
                    est = random_band(y[pair], treated, n_bootstrap, rng)
                else:
                    est = qini_with_interval(score, y[pair], treated, n_bootstrap, rng)
                rows.append({"outcome": outcome, "model": name, "arm": arm, **est})
                shares, q = qini_curve(score, y[pair], treated)
                curves[f"{outcome}|{name}|{arm}"] = q.tolist()
    curves["shares"] = shares.tolist()
    return rows, curves


def policy_results(
    test: pd.DataFrame, models: list[str], economics: Economics, n_bootstrap: int, reference: str
) -> list[dict]:
    t = test[TREATMENT_COL].to_numpy()
    reward = rewards(test[POLICY_OUTCOME], t, economics)
    policies = candidate_policies(test, models, economics)
    draws = bootstrap_policy_values(policies, t, reward, n_bootstrap, SEED)
    rows = []
    for name, arms in policies.items():
        diff = draws[name] - draws[reference]
        rows.append(
            {
                "policy": name,
                "value": policy_value(arms, t, reward),
                "ci_low": float(np.percentile(draws[name], 2.5)),
                "ci_high": float(np.percentile(draws[name], 97.5)),
                f"diff_vs_{reference}": float(np.mean(diff)),
                "diff_ci_low": float(np.percentile(diff, 2.5)),
                "diff_ci_high": float(np.percentile(diff, 97.5)),
                "share_mens": float(np.mean(arms == 1)),
                "share_womens": float(np.mean(arms == 2)),
                "share_none": float(np.mean(arms == 0)),
            }
        )
    return rows


def cost_sensitivity(test: pd.DataFrame, model: str, margin: float) -> list[dict]:
    """Test value of the model policy and of the uniform policies across email costs."""
    t = test[TREATMENT_COL].to_numpy()
    rows = []
    for cost in COST_GRID:
        economics = Economics(margin=margin, cost_per_email=cost)
        reward = rewards(test[POLICY_OUTCOME], t, economics)
        policies = candidate_policies(test, [model], economics)
        row = {"cost_per_email": cost}
        for name in ("no_email", "all_mens", "category_rule", f"model_{model}"):
            row[name] = policy_value(policies[name], t, reward)
        row["model_share_sent"] = float(np.mean(policies[f"model_{model}"] != 0))
        rows.append(row)
    return rows


def run(artifacts_dir: Path) -> dict:
    training = json.loads((artifacts_dir / "training.json").read_text())
    test = pd.read_parquet(artifacts_dir / "test_scores.parquet")
    economics = Economics(**training["economics"])
    n_bootstrap = training["settings"]["n_bootstrap"]
    models = training["models"]
    qini_rows, curves = qini_results(test, BASELINES + models, n_bootstrap)
    reference = "all_mens"
    summary = {
        "data": training["data"],
        "n_test": len(test),
        "economics": training["economics"],
        "selected_model": training["selected_model"],
        "selected_policy": training["selected_policy"],
        "validation_policy_values": training["validation_policy_values"],
        "qini": qini_rows,
        "policies": policy_results(test, models, economics, n_bootstrap, reference),
        "policy_reference": reference,
        "cost_sensitivity": cost_sensitivity(test, training["selected_model"], economics.margin),
    }
    (artifacts_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
    (artifacts_dir / "qini_curves.json").write_text(json.dumps(curves))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    args = parse_args(parser, __file__)
    summary = run(args.artifacts)
    print(pd.DataFrame(summary["qini"]).round(4).to_string(index=False))
    print(pd.DataFrame(summary["policies"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
