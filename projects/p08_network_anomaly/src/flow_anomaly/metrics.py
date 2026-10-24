"""Detection metrics: PR AUC, recall at a fixed false alarm rate, precision of the top alerts, and
bootstrap intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(normal_scores: np.ndarray, fpr: float) -> float:
    """Score above which the given share of normal flows lies; flows above it raise an alert."""
    return float(np.quantile(normal_scores, 1.0 - fpr, method="higher"))


def at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    alert = scores > threshold
    n_alerts = int(alert.sum())
    return {
        "recall": float(alert[y == 1].mean()) if (y == 1).any() else float("nan"),
        "fpr": float(alert[y == 0].mean()) if (y == 0).any() else float("nan"),
        "precision": float(y[alert].mean()) if n_alerts else float("nan"),
        "alert_share": float(alert.mean()),
    }


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return float(y[top].mean())


def ranking(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def bootstrap(
    y: np.ndarray, scores: np.ndarray, threshold: float, n: int, seed: int
) -> dict[str, tuple[float, float]]:
    """95 percent percentile intervals of PR AUC and recall at the threshold over resampled flows."""
    rng = np.random.default_rng(seed)
    pr, recall = [], []
    for _ in range(n):
        rows = rng.integers(0, len(y), len(y))
        yb, sb = y[rows], scores[rows]
        pr.append(average_precision_score(yb, sb))
        recall.append((sb[yb == 1] > threshold).mean())
    return {
        "pr_auc": tuple(np.percentile(pr, [2.5, 97.5]).tolist()),
        "recall": tuple(np.percentile(recall, [2.5, 97.5]).tolist()),
    }
