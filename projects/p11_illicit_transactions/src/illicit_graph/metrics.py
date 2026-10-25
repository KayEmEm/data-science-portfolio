"""Classification metrics for the illicit class: F1 at a threshold, the threshold with the best F1, and
intervals from resampled time steps."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "flagged": int(flagged.sum())}


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold with the highest F1 on these transactions (the validation steps)."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return float(thresholds[int(np.argmax(f1[:-1]))])


def ranking(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {"pr_auc": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores))}


def by_step(y: np.ndarray, scores: np.ndarray, steps: np.ndarray, threshold: float) -> list[dict]:
    rows = []
    for step in np.unique(steps):
        part = steps == step
        rows.append(
            {
                "time_step": int(step),
                "labelled": int(part.sum()),
                "illicit": int(y[part].sum()),
                **at_threshold(y[part], scores[part], threshold),
            }
        )
    return rows


def step_bootstrap(
    y: np.ndarray,
    steps: np.ndarray,
    statistic,
    n: int,
    seed: int,
) -> np.ndarray:
    """Values of ``statistic(rows)`` over ``n`` resamples of whole time steps with replacement.

    Transactions of one step share their graph and their period, so steps rather than transactions
    are the units that vary independently.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(steps)
    rows_of = {step: np.flatnonzero(steps == step) for step in unique}
    out = []
    for _ in range(n):
        chosen = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([rows_of[s] for s in chosen])
        if y[rows].sum() == 0:
            continue
        out.append(statistic(rows))
    return np.asarray(out)


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5], axis=0)
    return float(low), float(high)
