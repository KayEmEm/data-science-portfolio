"""Evaluation helpers shared across classification projects."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return ROC AUC, PR AUC (average precision) and Brier score."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
