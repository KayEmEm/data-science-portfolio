import numpy as np

from loan_default.config import CostAssumptions
from loan_default.fairness import group_report
from loan_default.threshold import best_threshold, break_even_threshold, profit_curve


def test_break_even_threshold():
    assert break_even_threshold(CostAssumptions(loss_given_default=0.45, margin=0.10)) == 0.10 / 0.55


def test_profit_curve_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.10, 0.60, 0.90])
    exposure = np.array([100.0, 100.0, 100.0, 100.0])
    curve = profit_curve(y, p, exposure, CostAssumptions(loss_given_default=0.5, margin=0.1))
    # Approving only the two good borrowers earns 2 * 10.
    assert curve["profit"].max() == 20.0
    t = best_threshold(curve)
    assert 0.10 < t <= 0.60
    assert curve.loc[curve["threshold"] == 1.0, "approval_rate"].item() == 1.0


def test_group_report_rates():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.2, 0.3, 0.7, 0.1])
    age = np.array([25, 26, 45, 46, 47, 65])
    report = group_report(y, p, age, threshold=0.5).set_index("age_band")
    assert report.loc["under 30", "approval_rate"] == 0.5
    assert report.loc["40 to 49", "rejected_good_rate"] == 0.5
    assert report.loc["40 to 49", "approved_bad_rate"] == 1.0
    assert report["n"].sum() == 6
