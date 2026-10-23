"""Group metrics across age bands at the chosen threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["under 30", "30 to 39", "40 to 49", "50 to 59", "60 and over"]


def age_band(age_years: pd.Series) -> pd.Series:
    return pd.cut(age_years, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def group_report(
    y_true: np.ndarray, p_default: np.ndarray, age_years: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Per age band: size, default rate, approval rate, error rates and AUC.

    ``rejected_good_rate`` is the share of borrowers who repaid but were rejected.
    ``approved_bad_rate`` is the share of defaulters who were approved.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(p_default), "band": age_band(pd.Series(age_years))}
    )
    frame["approved"] = frame["p"] < threshold
    rows = []
    for band, g in frame.groupby("band", observed=True):
        good, bad = g[g["y"] == 0], g[g["y"] == 1]
        rows.append(
            {
                "age_band": str(band),
                "n": len(g),
                "default_rate": g["y"].mean(),
                "approval_rate": g["approved"].mean(),
                "rejected_good_rate": (~good["approved"]).mean() if len(good) else np.nan,
                "approved_bad_rate": bad["approved"].mean() if len(bad) else np.nan,
                "mean_predicted_pd": g["p"].mean(),
                "roc_auc": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    overall = frame["approved"].mean()
    report["approval_ratio_to_overall"] = report["approval_rate"] / overall if overall else np.nan
    return report
