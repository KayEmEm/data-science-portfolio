"""Write the comparison table, threshold figure, calibration figure, SHAP figure and fairness table."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from portfolio_common.plotting import apply_style
from sklearn.calibration import calibration_curve

from loan_default.config import (
    AGE_COL,
    ARTIFACTS_DIR,
    EXPOSURE_COL,
    REPORTS_DIR,
    SEED,
    TARGET_COL,
    CostAssumptions,
)
from loan_default.fairness import group_report
from loan_default.modeling import FittedModel
from loan_default.threshold import profit_curve

LABELS = {
    "logistic_regression": "Logistic regression (baseline)",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def comparison_table(summary: dict) -> pd.DataFrame:
    rows = []
    for name, r in summary["models"].items():
        rows.append(
            {
                "model": LABELS[name],
                "CV ROC AUC": r["cv_raw"]["roc_auc"],
                "test ROC AUC": r["test_calibrated"]["roc_auc"],
                "test PR AUC": r["test_calibrated"]["pr_auc"],
                "test Brier (raw)": r["test_raw"]["brier"],
                "test Brier (calibrated)": r["test_calibrated"]["brier"],
                "threshold": r["threshold"],
                "test approval rate": r["test_approval_rate"],
                "test profit": r["test_profit_at_threshold"],
            }
        )
    return pd.DataFrame(rows)


def threshold_figure(test: pd.DataFrame, name: str, summary: dict, out: Path) -> None:
    costs = CostAssumptions(**summary["costs"])
    curve = profit_curve(test[TARGET_COL], test[f"p_{name}"], test[EXPOSURE_COL], costs)
    chosen = summary["models"][name]["threshold"]
    fig, ax = plt.subplots()
    ax.plot(curve["threshold"], curve["profit"] / 1e6, label="profit on test split (millions)")
    ax.axvline(chosen, linestyle="--", color="#3d405b", label=f"chosen threshold {chosen:.3f}")
    ax.axvline(
        summary["break_even_threshold"],
        linestyle=":",
        color="#e07a5f",
        label=f"break even rule {summary['break_even_threshold']:.3f}",
    )
    ax.set_xlabel("approve when predicted probability of default is below")
    ax.set_ylabel("profit (millions of NT dollars)")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["approval_rate"], color="#81b29a", alpha=0.8, label="approval rate")
    ax2.set_ylabel("approval rate")
    ax2.grid(False)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="lower right", fontsize=8)
    ax.set_title(f"Threshold analysis: {LABELS[name]}")
    fig.savefig(out)
    plt.close(fig)


def calibration_figure(test: pd.DataFrame, name: str, out: Path) -> None:
    fig, ax = plt.subplots()
    for column, label in ((f"p_raw_{name}", "raw"), (f"p_{name}", "isotonic calibrated")):
        frac, mean = calibration_curve(test[TARGET_COL], test[column], n_bins=15, strategy="quantile")
        ax.plot(mean, frac, marker="o", markersize=3, label=label)
    top = float(max(test[f"p_raw_{name}"].quantile(0.99), test[f"p_{name}"].quantile(0.99)))
    ax.plot([0, top], [0, top], color="grey", linestyle=":", label="perfect calibration")
    ax.set_xlabel("mean predicted probability of default")
    ax.set_ylabel("observed default rate")
    ax.set_title(f"Calibration on test split: {LABELS[name]}")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def tree_shap_values(fm: FittedModel, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values from the boosting library itself (last column is the bias)."""
    if fm.name == "lightgbm":
        return fm.estimator.booster_.predict(X, pred_contrib=True)[:, :-1]
    if fm.name == "xgboost":
        dmat = xgb.DMatrix(X, enable_categorical=True)
        return fm.estimator.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
    raise ValueError("SHAP values are computed for tree models only")


def explanation_figure(fm: FittedModel, sample: pd.DataFrame, out: Path) -> None:
    X = fm.preprocessor.transform(sample)
    if fm.name == "logistic_regression":
        clf = fm.estimator.named_steps["clf"]
        names = fm.estimator.named_steps["prep"].get_feature_names_out()
        coef = pd.Series(clf.coef_[0], index=names).sort_values(key=np.abs).tail(20)
        fig, ax = plt.subplots(figsize=(7, 6))
        coef.plot.barh(ax=ax)
        ax.set_title("Largest standardized coefficients (logistic regression)")
        fig.savefig(out)
        plt.close(fig)
        return
    values = tree_shap_values(fm, X)
    display = X.apply(
        lambda s: s.cat.codes.replace(-1, np.nan) if isinstance(s.dtype, pd.CategoricalDtype) else s
    )
    shap.summary_plot(
        values, display, max_display=20, show=False, plot_size=(8, 7), rng=np.random.default_rng(SEED)
    )
    plt.title(f"SHAP values on {len(X):,} test clients: {LABELS[fm.name]}")
    plt.savefig(out, bbox_inches="tight")
    plt.close("all")


def run(artifacts_dir: Path, reports_dir: Path) -> None:
    apply_style()
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((artifacts_dir / "metrics.json").read_text())
    test = pd.read_parquet(artifacts_dir / "test_predictions.parquet")
    bundle = joblib.load(artifacts_dir / "model.joblib")
    name = summary["selected_model"]

    table = comparison_table(summary)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison.md").write_text(table.to_markdown(index=False, floatfmt=".4f") + "\n")

    fairness = group_report(test[TARGET_COL], test[f"p_{name}"], test[AGE_COL], bundle["threshold"])
    fairness.to_csv(reports_dir / "fairness_age.csv", index=False)
    (reports_dir / "fairness_age.md").write_text(fairness.to_markdown(index=False, floatfmt=".4f") + "\n")

    threshold_figure(test, name, summary, figures / "threshold_analysis.png")
    calibration_figure(test, name, figures / "calibration.png")
    explanation_figure(
        bundle["model"], pd.read_parquet(artifacts_dir / "shap_sample.parquet"), figures / "explanations.png"
    )
    (reports_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"data: {summary['data']}; selected model: {LABELS[name]}")
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run(args.artifacts, args.reports)


if __name__ == "__main__":
    main()
