"""Detection metrics: PR AUC, recall at a fixed false alarm rate, precision of the top alerts, and
bootstrap intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


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


def pr_auc_at_prevalence(y: np.ndarray, scores: np.ndarray, prevalence: float) -> float:
    """Average precision that the same ranking would reach if attacks made up ``prevalence`` of the flows.

    Recall and false alarm rate do not depend on the attack share, so the precision at each threshold
    is recomputed from them as recall * p / (recall * p + fpr * (1 - p)).
    """
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))
    precision = np.nan_to_num(precision, nan=1.0)
    return float(np.sum(np.diff(tpr) * precision[1:]))


def ranking(y: np.ndarray, scores: np.ndarray, prevalence: float | None = None) -> dict[str, float]:
    out = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }
    if prevalence is not None:
        out["pr_auc_rescaled"] = pr_auc_at_prevalence(y, scores, prevalence)
    return out


def bootstrap(
    y: np.ndarray, scores: np.ndarray, threshold: float, n: int, seed: int, prevalence: float | None = None
) -> dict[str, tuple[float, float]]:
    """95 percent percentile intervals of PR AUC (also rescaled to ``prevalence``) and recall at the
    threshold over resampled flows."""
    rng = np.random.default_rng(seed)
    pr, rescaled, recall = [], [], []
    for _ in range(n):
        rows = rng.integers(0, len(y), len(y))
        yb, sb = y[rows], scores[rows]
        pr.append(average_precision_score(yb, sb))
        if prevalence is not None:
            rescaled.append(pr_auc_at_prevalence(yb, sb, prevalence))
        recall.append((sb[yb == 1] > threshold).mean())
    out = {
        "pr_auc": tuple(np.percentile(pr, [2.5, 97.5]).tolist()),
        "recall": tuple(np.percentile(recall, [2.5, 97.5]).tolist()),
    }
    if prevalence is not None:
        out["pr_auc_rescaled"] = tuple(np.percentile(rescaled, [2.5, 97.5]).tolist())
    return out
