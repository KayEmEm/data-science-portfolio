"""Train the baseline and the two boosted models, calibrate them, pick a threshold.

Model selection and the threshold use only the training split (out of fold
predictions). The held out test split is scored once, for reporting.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from portfolio_common.evaluation import binary_metrics
from portfolio_common.seed import set_seed
from sklearn.model_selection import train_test_split

from loan_default.config import (
    ARTIFACTS_DIR,
    FEATURES_PATH,
    ID_COL,
    SEED,
    TARGET_COL,
    CostAssumptions,
    RunSettings,
)
from loan_default.modeling import MODEL_NAMES, fit_with_cv
from loan_default.threshold import best_threshold, break_even_threshold, profit_curve

EXPERIMENT = "p01_loan_default"
REPORT_COLUMNS = [ID_COL, TARGET_COL, "AMT_CREDIT", "APP_AGE_YEARS"]


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def example_request(row: pd.Series) -> dict:
    features = {k: _json_value(v) for k, v in row.drop([ID_COL, TARGET_COL]).items() if pd.notna(v)}
    return {"features": features}


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{artifacts_dir / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def run(
    features_path: Path,
    artifacts_dir: Path,
    settings: RunSettings,
    costs: CostAssumptions,
    data_label: str,
    models: list[str],
) -> dict:
    set_seed(SEED)
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(artifacts_dir)
    frame = pd.read_parquet(features_path)
    train, test = train_test_split(
        frame, test_size=settings.test_size, stratify=frame[TARGET_COL], random_state=SEED
    )
    y_train, y_test = train[TARGET_COL].to_numpy(), test[TARGET_COL].to_numpy()

    oof_out = train[REPORT_COLUMNS].reset_index(drop=True)
    test_out = test[REPORT_COLUMNS].reset_index(drop=True)
    summary = {
        "data": data_label,
        "n_train": len(train),
        "n_test": len(test),
        "train_default_rate": float(y_train.mean()),
        "costs": asdict(costs),
        "break_even_threshold": break_even_threshold(costs),
        "models": {},
    }
    fitted = {}
    for name in models:
        with mlflow.start_run(run_name=name):
            fm = fit_with_cv(name, train, settings)
            oof_cal = fm.calibrator.predict(fm.oof_raw)
            test_raw = fm.predict_raw(test)
            test_cal = fm.calibrator.predict(test_raw)
            curve = profit_curve(y_train, oof_cal, train["AMT_CREDIT"], costs)
            t = best_threshold(curve)
            test_curve = profit_curve(y_test, test_cal, test["AMT_CREDIT"], costs)
            at_t = test_curve.loc[np.isclose(test_curve["threshold"], t)].iloc[0]
            result = {
                "cv_raw": binary_metrics(y_train, fm.oof_raw),
                "test_raw": binary_metrics(y_test, test_raw),
                "test_calibrated": binary_metrics(y_test, test_cal),
                "threshold": t,
                "cv_profit_at_threshold": float(curve["profit"].max()),
                "test_profit_at_threshold": float(at_t["profit"]),
                "test_approval_rate": float(at_t["approval_rate"]),
                "test_bad_rate_approved": float(at_t["bad_rate_approved"]),
                "n_estimators": fm.n_estimators,
            }
            summary["models"][name] = result
            mlflow.set_tags({"data": data_label})
            mlflow.log_params(
                {"model": name, "n_folds": settings.n_folds, "n_estimators": fm.n_estimators, **asdict(costs)}
            )
            mlflow.log_metrics(
                {
                    f"{split}_{k}": v
                    for split in ("cv_raw", "test_raw", "test_calibrated")
                    for k, v in result[split].items()
                }
                | {"threshold": t, "test_profit_at_threshold": result["test_profit_at_threshold"]}
            )
            oof_out[f"p_{name}"] = oof_cal
            test_out[f"p_{name}"] = test_cal
            test_out[f"p_raw_{name}"] = test_raw
            fitted[name] = fm

    # Selection uses training data only: the highest out of fold profit.
    selected = max(models, key=lambda m: summary["models"][m]["cv_profit_at_threshold"])
    summary["selected_model"] = selected
    joblib.dump(
        {
            "model": fitted[selected],
            "threshold": summary["models"][selected]["threshold"],
            "costs": asdict(costs),
            "data": data_label,
        },
        artifacts_dir / "model.joblib",
    )
    oof_out.to_parquet(artifacts_dir / "oof_predictions.parquet", index=False)
    test_out.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
    test.sample(min(settings.shap_sample, len(test)), random_state=SEED).to_parquet(
        artifacts_dir / "shap_sample.parquet", index=False
    )
    (artifacts_dir / "example_request.json").write_text(json.dumps(example_request(test.iloc[0]), indent=2))
    (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="small settings for a smoke run")
    parser.add_argument("--data-label", default="home-credit", help="recorded with every output")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--lgd", type=float, default=CostAssumptions.loss_given_default)
    parser.add_argument("--margin", type=float, default=CostAssumptions.margin)
    args = parser.parse_args()
    settings = RunSettings.quick() if args.quick else RunSettings()
    costs = CostAssumptions(loss_given_default=args.lgd, margin=args.margin)
    summary = run(args.features, args.artifacts, settings, costs, args.data_label, args.models)
    print(json.dumps({m: r["test_calibrated"] for m, r in summary["models"].items()}, indent=2))
    print(f"selected model: {summary['selected_model']}")


if __name__ == "__main__":
    main()
