"""Approval threshold from a cost matrix.

A borrower is approved when the predicted probability of default is below the
threshold. With calibrated probabilities, approving is worth it when
``(1 - p) * margin > p * loss_given_default``, which gives the break even
threshold ``margin / (margin + loss_given_default)``. The empirical threshold
below checks that rule against realised outcomes.
"""

import numpy as np
import pandas as pd

from loan_default.config import CostAssumptions

GRID = np.round(np.arange(0.0, 1.0001, 0.005), 4)


def break_even_threshold(costs: CostAssumptions) -> float:
    return costs.margin / (costs.margin + costs.loss_given_default)


def profit_curve(
    y_true: np.ndarray, p_default: np.ndarray, exposure: np.ndarray, costs: CostAssumptions
) -> pd.DataFrame:
    """Realised profit, approval rate and bad rate among approved, per threshold."""
    y_true = np.asarray(y_true)
    p_default = np.asarray(p_default)
    exposure = np.asarray(exposure, dtype=float)
    gain = np.where(y_true == 0, costs.margin * exposure, -costs.loss_given_default * exposure)
    rows = []
    for t in GRID:
        approved = p_default < t
        n_approved = int(approved.sum())
        rows.append(
            {
                "threshold": float(t),
                "profit": float(gain[approved].sum()),
                "approval_rate": n_approved / len(y_true),
                "bad_rate_approved": float(y_true[approved].mean()) if n_approved else 0.0,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])
