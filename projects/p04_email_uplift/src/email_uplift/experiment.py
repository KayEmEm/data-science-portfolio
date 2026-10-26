"""Readout of the randomized experiment: checks, effects, power and variance reduction.

Each email arm is compared with the control arm on visits, conversions and
spend in the two weeks after the send. The checks come first (sample ratio
and covariate balance), because an effect estimate is only as good as the
randomization behind it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from portfolio_common.config import parse_args
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower

from email_uplift.config import (
    ARM_COL,
    ARM_LABELS,
    ARMS,
    ARTIFACTS_DIR,
    CONTROL,
    COVARIATES,
    CUSTOMERS_PATH,
    OUTCOMES,
    PRE_PERIOD_COL,
)
from email_uplift.data import feature_matrix, load

ALPHA = 0.05
POWER = 0.80


def sample_ratio_check(frame: pd.DataFrame) -> dict:
    """Chi square test of the arm counts against the planned equal split."""
    counts = frame[ARM_COL].value_counts().reindex([CONTROL, *ARMS])
    result = stats.chisquare(counts.to_numpy())
    return {
        "counts": {ARM_LABELS[k]: int(v) for k, v in counts.items()},
        "chi2": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference of each pre treatment feature, arm against control.

    An absolute value below 0.1 is the usual threshold for adequate balance.
    """
    features = feature_matrix(frame)
    control = features[frame[ARM_COL] == CONTROL]
    rows = []
    for col in features.columns:
        row = {"feature": col, "control_mean": control[col].mean()}
        for arm in ARMS:
            treated = features.loc[frame[ARM_COL] == arm, col]
            pooled_sd = np.sqrt((treated.var() + control[col].var()) / 2)
            smd = (treated.mean() - control[col].mean()) / pooled_sd if pooled_sd > 0 else 0.0
            row[f"smd_{ARM_LABELS[arm]}"] = smd
        rows.append(row)
    return pd.DataFrame(rows)


def difference_in_means(y_treated: np.ndarray, y_control: np.ndarray, alpha: float = ALPHA) -> dict:
    """Difference in means with a Welch standard error and a normal confidence interval."""
    y_treated, y_control = np.asarray(y_treated, float), np.asarray(y_control, float)
    diff = y_treated.mean() - y_control.mean()
    se = np.sqrt(y_treated.var(ddof=1) / len(y_treated) + y_control.var(ddof=1) / len(y_control))
    z = stats.norm.ppf(1 - alpha / 2)
    p_value = stats.ttest_ind(y_treated, y_control, equal_var=False).pvalue
    control_mean = y_control.mean()
    return {
        "treated_mean": float(y_treated.mean()),
        "control_mean": float(control_mean),
        "effect": float(diff),
        "se": float(se),
        "ci_low": float(diff - z * se),
        "ci_high": float(diff + z * se),
        "relative_lift": float(diff / control_mean) if control_mean else float("nan"),
        "p_value": float(p_value),
    }


def effects_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Every arm against control on every outcome, with Holm adjusted p values."""
    control = frame[frame[ARM_COL] == CONTROL]
    rows = []
    for arm in ARMS:
        treated = frame[frame[ARM_COL] == arm]
        for outcome in OUTCOMES:
            est = difference_in_means(treated[outcome], control[outcome])
            rows.append({"arm": ARM_LABELS[arm], "outcome": outcome, **est})
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p_value"], alpha=ALPHA, method="holm")[1]
    return table


def minimum_detectable_effect(sd: float, n_treated: int, n_control: int) -> float:
    """Smallest true difference detected with probability POWER at level ALPHA (two sided)."""
    d = NormalIndPower().solve_power(
        nobs1=n_treated, ratio=n_control / n_treated, alpha=ALPHA, power=POWER, alternative="two-sided"
    )
    return float(d * sd)


def required_sample_size(effect: float, sd: float) -> int:
    """Customers per arm needed to detect ``effect`` with probability POWER, equal arms."""
    if effect == 0:
        return -1
    n = NormalIndPower().solve_power(effect_size=abs(effect) / sd, alpha=ALPHA, power=POWER, ratio=1.0)
    return int(np.ceil(n))


def power_table(frame: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Design view: what the test could detect, and what each observed effect would need."""
    control = frame[frame[ARM_COL] == CONTROL]
    rows = []
    for arm in ARMS:
        n_treated = int((frame[ARM_COL] == arm).sum())
        for outcome in OUTCOMES:
            sd = float(control[outcome].std(ddof=1))
            match = (effects["arm"] == ARM_LABELS[arm]) & (effects["outcome"] == outcome)
            observed = effects.loc[match, "effect"].iloc[0]
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
                    "outcome": outcome,
                    "control_sd": sd,
                    "n_per_arm": n_treated,
                    "mde": minimum_detectable_effect(sd, n_treated, len(control)),
                    "observed_effect": float(observed),
                    "n_needed_for_observed": required_sample_size(observed, sd),
                }
            )
    return pd.DataFrame(rows)


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: subtract the part of the outcome that the pre period covariate predicts.

    theta is estimated on all arms pooled. Because x is measured before
    randomization, the adjustment leaves the expected difference unchanged and
    removes the share corr(y, x)^2 of the outcome variance.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def regression_adjusted(frame: pd.DataFrame, outcome: str, arm: str) -> dict:
    """OLS of the outcome on the arm indicator and centered covariates, with HC2 errors.

    Covariates are interacted with the arm indicator (Lin, 2013), so the
    adjustment cannot make the precision worse in large samples.
    """
    sub = frame[frame[ARM_COL].isin([CONTROL, arm])]
    features = feature_matrix(sub).drop(columns=["history"])
    centered = features - features.mean()
    data = centered.assign(y=sub[outcome].to_numpy(), t=(sub[ARM_COL] == arm).astype(float).to_numpy())
    terms = " + ".join(f"t * {c}" for c in centered.columns)
    fit = smf.ols(f"y ~ {terms}", data=data).fit(cov_type="HC2")
    low, high = fit.conf_int(alpha=ALPHA).loc["t"]
    return {"effect": float(fit.params["t"]), "se": float(fit.bse["t"]), "ci_low": low, "ci_high": high}


def variance_reduction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Confidence interval width without adjustment, with CUPED and with regression adjustment."""
    control_mask = frame[ARM_COL] == CONTROL
    rows = []
    for outcome in OUTCOMES:
        adjusted, theta = cuped_adjust(frame[outcome], frame[PRE_PERIOD_COL])
        for arm in ARMS:
            arm_mask = frame[ARM_COL] == arm
            raw = difference_in_means(frame.loc[arm_mask, outcome], frame.loc[control_mask, outcome])
            cuped = difference_in_means(adjusted[arm_mask.to_numpy()], adjusted[control_mask.to_numpy()])
            reg = regression_adjusted(frame, outcome, arm)
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
                    "outcome": outcome,
                    "theta": theta,
                    "corr_with_pre_period": float(np.corrcoef(frame[outcome], frame[PRE_PERIOD_COL])[0, 1]),
                    "effect_raw": raw["effect"],
                    "se_raw": raw["se"],
                    "effect_cuped": cuped["effect"],
                    "se_cuped": cuped["se"],
                    "effect_regression": reg["effect"],
                    "se_regression": reg["se"],
                    "variance_reduction_cuped": 1 - (cuped["se"] / raw["se"]) ** 2,
                    "variance_reduction_regression": 1 - (reg["se"] / raw["se"]) ** 2,
                }
            )
    return pd.DataFrame(rows)


def run(customers_path: Path, artifacts_dir: Path, data_label: str) -> dict:
    frame = load(customers_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    effects = effects_table(frame)
    balance = balance_table(frame)
    smd_cols = [c for c in balance.columns if c.startswith("smd_")]
    summary = {
        "data": data_label,
        "n_customers": len(frame),
        "covariates": COVARIATES,
        "sample_ratio": sample_ratio_check(frame),
        "max_abs_smd": float(balance[smd_cols].abs().max().max()),
        "balance": balance.to_dict(orient="records"),
        "effects": effects.to_dict(orient="records"),
        "power": power_table(frame, effects).to_dict(orient="records"),
        "variance_reduction": variance_reduction_table(frame).to_dict(orient="records"),
        "alpha": ALPHA,
        "power_target": POWER,
    }
    (artifacts_dir / "experiment.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=Path, default=CUSTOMERS_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="hillstrom-2008", help="recorded with every output")
    args = parse_args(parser, __file__)
    summary = run(args.customers, args.artifacts, args.data_label)
    print(f"sample ratio p value {summary['sample_ratio']['p_value']:.3f}")
    print(pd.DataFrame(summary["effects"])[["arm", "outcome", "effect", "ci_low", "ci_high", "p_holm"]])


if __name__ == "__main__":
    main()
