"""Score every detector of every experiment on the validation and test flows.

The alarm threshold of a detector is set on the validation flows, so that 1 percent of normal
validation flows raise an alert; the test flows are then scored at that threshold. A second
threshold uses no labels at all: the score above which 1 percent of the training sample lies.

Outputs under ``artifacts/evaluation/``:

* ``summary.csv``: per experiment and detector, validation PR AUC, test ROC AUC and PR AUC, recall,
  false alarm rate and precision at the threshold, precision of the top 100 and 1000 alerts, and the
  same at the label free threshold; 95 percent intervals for the main experiment;
* ``by_type.csv``: test recall at the threshold per attack type (main experiment);
* ``selection.json``: the unsupervised detector with the best validation PR AUC.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from flow_anomaly import detectors
from flow_anomaly.config import ALERT_BUDGETS, ARTIFACTS_DIR, BOOTSTRAP, DATA_DIR, NORMAL, SEED, TARGET_FPR
from flow_anomaly.data import load
from flow_anomaly.metrics import at_threshold, bootstrap, precision_at_k, ranking, threshold_at_fpr
from flow_anomaly.train import experiments

EXPERIMENT = "p08_network_anomaly"


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def scores(artifacts: Path, experiment: str, detector: str, part: str) -> np.ndarray:
    return np.load(artifacts / experiment / "scores" / f"{detector}_{part}.npy")


def evaluate_detector(
    artifacts: Path, experiment: str, detector: str, labels: dict, n_bootstrap: int
) -> tuple[dict, pd.DataFrame]:
    s_val = scores(artifacts, experiment, detector, "validation")
    s_test = scores(artifacts, experiment, detector, "test")
    s_fit = scores(artifacts, experiment, detector, "fit")
    y_val, y_test, cat_test = labels["y_val"], labels["y_test"], labels["cat_test"]
    threshold = threshold_at_fpr(s_val[y_val == 0], TARGET_FPR)
    label_free = float(np.quantile(s_fit, 1.0 - TARGET_FPR, method="higher"))
    row = {
        "experiment": experiment,
        "detector": detector,
        "validation_pr_auc": ranking(y_val, s_val)["pr_auc"],
        **{f"test_{k}": v for k, v in ranking(y_test, s_test).items()},
        **{f"test_{k}": v for k, v in at_threshold(y_test, s_test, threshold).items()},
        **{f"test_precision@{k}": precision_at_k(y_test, s_test, k) for k in ALERT_BUDGETS},
        **{f"label_free_{k}": v for k, v in at_threshold(y_test, s_test, label_free).items()},
    }
    if n_bootstrap:
        intervals = bootstrap(y_test, s_test, threshold, n_bootstrap, SEED)
        row["test_pr_auc_low"], row["test_pr_auc_high"] = intervals["pr_auc"]
        row["test_recall_low"], row["test_recall_high"] = intervals["recall"]
    alert = s_test > threshold
    by_type = (
        pd.DataFrame({"attack_cat": cat_test, "alert": alert})
        .groupby("attack_cat")["alert"]
        .agg(flows="size", recall="mean")
        .reset_index()
    )
    by_type.insert(0, "detector", detector)
    return row, by_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="UNSW-NB15")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    flows = load(args.data_dir)
    split = flows["split"].to_numpy()
    labels = {
        "y_val": flows.loc[split == "validation", "label"].to_numpy(),
        "y_test": flows.loc[split == "test", "label"].to_numpy(),
        "cat_test": flows.loc[split == "test", "attack_cat"].to_numpy(),
    }
    out = args.artifacts / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows, by_type = [], []
    for experiment, setup in experiments().items():
        for detector in setup["detectors"]:
            if not (args.artifacts / experiment / "scores" / f"{detector}_test.npy").exists():
                print(f"{experiment}/{detector}: no scores, skipped")
                continue
            main_run = experiment == "main"
            row, types = evaluate_detector(
                args.artifacts, experiment, detector, labels, args.bootstrap if main_run else 0
            )
            row["attack_share"] = setup["attack_share"]
            rows.append(row)
            if main_run:
                by_type.append(types)
            print(
                f"{experiment:<20} {detector:<12} val PR AUC {row['validation_pr_auc']:.3f}  "
                f"test PR AUC {row['test_pr_auc']:.3f}  "
                f"recall at {TARGET_FPR:.0%} FPR {row['test_recall']:.3f}"
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    types = pd.concat(by_type, ignore_index=True)
    types = types.loc[types["attack_cat"] != NORMAL]
    types.to_csv(out / "by_type.csv", index=False)

    main = summary.loc[(summary["experiment"] == "main") & summary["detector"].isin(detectors.UNSUPERVISED)]
    best = main.sort_values("validation_pr_auc", ascending=False).iloc[0]
    selection = {
        "detector": best["detector"],
        "label": detectors.SPECS[best["detector"]].label,
        "validation_pr_auc": float(best["validation_pr_auc"]),
        "data": args.data_label,
        "test_attack_share": float(labels["y_test"].mean()),
        "test_flows": int(len(labels["y_test"])),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"selected on validation PR AUC: {selection['label']}")

    setup_mlflow(args.artifacts)
    for row in rows:
        with mlflow.start_run(run_name=f"{row['experiment']}_{row['detector']}"):
            mlflow.set_tags(
                {"data": args.data_label, "experiment": row["experiment"], "detector": row["detector"]}
            )
            mlflow.log_metrics(
                {
                    k.replace("@", "_at_"): float(v)
                    for k, v in row.items()
                    if isinstance(v, float) and np.isfinite(v)
                }
            )


if __name__ == "__main__":
    main()
