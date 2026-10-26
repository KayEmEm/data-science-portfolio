"""Score every model on the labelled transactions of the validation and test steps.

The decision threshold of a model is the one with the best illicit F1 on the validation steps; the
test steps are then scored once at that threshold, and also at 0.5 (the convention of the published
benchmark). Intervals come from resampling the 15 test steps with replacement.

Outputs under ``artifacts/evaluation/``:

* ``summary.csv``: per model, validation F1 and threshold, test F1, precision and recall at that
  threshold and at 0.5, PR AUC, ROC AUC, F1 before and from the dark market shutdown, 95 percent
  intervals, and for the graph networks the range of test F1 over seeds;
* ``by_step.csv``: test F1, precision and recall per model and time step;
* ``paired.csv``: test F1 of each model minus that of the feature only LightGBM model, with an
  interval from the same resampled steps;
* ``selection.json``: the model with the best validation F1.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from illicit_graph.config import ARTIFACTS_DIR, BOOTSTRAP, DATA_DIR, ILLICIT, SEED, SHUTDOWN_STEP
from illicit_graph.data import load
from illicit_graph.metrics import at_threshold, best_f1_threshold, by_step, interval, ranking, step_bootstrap
from illicit_graph.train import GNN_KINDS, MODELS, masks

EXPERIMENT = "p11_illicit_transactions"
REFERENCE = "lightgbm_all"


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def seed_scores(out: Path) -> list[np.ndarray]:
    return [np.load(p) for p in sorted(out.glob("scores_seed*.npy"))]


def evaluate_model(
    name: str, scores: np.ndarray, parts: dict, n_bootstrap: int, artifacts: Path
) -> tuple[dict, list]:
    yv, sv = parts["y_val"], scores[parts["validation"]]
    yt, st, steps = parts["y_test"], scores[parts["test"]], parts["steps_test"]
    threshold = best_f1_threshold(yv, sv)
    before, after = steps < SHUTDOWN_STEP, steps >= SHUTDOWN_STEP
    row = {
        "model": name,
        "label": MODELS[name]["label"],
        "threshold": threshold,
        "validation_f1": at_threshold(yv, sv, threshold)["f1"],
        **{f"validation_{k}": v for k, v in ranking(yv, sv).items()},
        **{f"test_{k}": v for k, v in at_threshold(yt, st, threshold).items()},
        **{f"test_{k}_at_half": v for k, v in at_threshold(yt, st, 0.5).items() if k != "flagged"},
        **{f"test_{k}": v for k, v in ranking(yt, st).items()},
        "test_f1_before_shutdown": at_threshold(yt[before], st[before], threshold)["f1"],
        "test_f1_from_shutdown": at_threshold(yt[after], st[after], threshold)["f1"],
    }
    if n_bootstrap:
        f1 = step_bootstrap(
            yt, steps, lambda r: at_threshold(yt[r], st[r], threshold)["f1"], n_bootstrap, SEED
        )
        pr = step_bootstrap(yt, steps, lambda r: ranking(yt[r], st[r])["pr_auc"], n_bootstrap, SEED)
        row["test_f1_low"], row["test_f1_high"] = interval(f1)
        row["test_pr_auc_low"], row["test_pr_auc_high"] = interval(pr)
    if MODELS[name]["kind"] in GNN_KINDS:
        per_seed = []
        for prob in seed_scores(artifacts / "models" / name):
            t = best_f1_threshold(yv, prob[parts["validation"]])
            per_seed.append(at_threshold(yt, prob[parts["test"]], t)["f1"])
        row["test_f1_seed_min"], row["test_f1_seed_max"] = min(per_seed), max(per_seed)
        row["seeds"] = len(per_seed)
    steps_rows = [{"model": name, **r} for r in by_step(yt, st, steps, threshold)]
    return row, steps_rows


def paired(scores: dict, thresholds: dict, parts: dict, n_bootstrap: int) -> pd.DataFrame:
    yt, steps = parts["y_test"], parts["steps_test"]
    ref = scores[REFERENCE][parts["test"]]
    rows = []
    for name, prob in scores.items():
        if name == REFERENCE:
            continue
        st = prob[parts["test"]]

        def diff(r, st=st, name=name):
            return (
                at_threshold(yt[r], st[r], thresholds[name])["f1"]
                - at_threshold(yt[r], ref[r], thresholds[REFERENCE])["f1"]
            )

        values = step_bootstrap(yt, steps, diff, n_bootstrap, SEED) if n_bootstrap else np.array([np.nan])
        all_rows = np.arange(len(yt))
        low, high = interval(values) if n_bootstrap else (np.nan, np.nan)
        rows.append(
            {"model": name, "reference": REFERENCE, "f1_difference": diff(all_rows), "low": low, "high": high}
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="Elliptic")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parse_args(parser, __file__)
    nodes, _ = load(args.data_dir)
    m = masks(nodes)
    y = (nodes["label"].to_numpy() == ILLICIT).astype(int)
    steps = nodes["time_step"].to_numpy()
    parts = {
        "validation": m["validation"],
        "test": m["test"],
        "y_val": y[m["validation"]],
        "y_test": y[m["test"]],
        "steps_test": steps[m["test"]],
    }
    out = args.artifacts / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows, step_rows, scores = [], [], {}
    for name in MODELS:
        path = args.artifacts / "models" / name / "scores.npy"
        if not path.exists():
            print(f"{name}: no scores, skipped")
            continue
        scores[name] = np.load(path)
        row, s_rows = evaluate_model(name, scores[name], parts, args.bootstrap, args.artifacts)
        rows.append(row)
        step_rows.extend(s_rows)
        print(
            f"{name:<20} validation F1 {row['validation_f1']:.3f}  test F1 {row['test_f1']:.3f}  "
            f"precision {row['test_precision']:.3f}  recall {row['test_recall']:.3f}  "
            f"F1 from step {SHUTDOWN_STEP} {row['test_f1_from_shutdown']:.3f}"
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(step_rows).to_csv(out / "by_step.csv", index=False)
    thresholds = dict(zip(summary["model"], summary["threshold"], strict=True))
    if REFERENCE in scores:
        paired(scores, thresholds, parts, args.bootstrap).to_csv(out / "paired.csv", index=False)
    best = summary.sort_values("validation_f1", ascending=False).iloc[0]
    selection = {
        "model": best["model"],
        "label": best["label"],
        "validation_f1": float(best["validation_f1"]),
        "threshold": float(best["threshold"]),
        "data": args.data_label,
        "test_labelled": int(len(parts["y_test"])),
        "test_illicit": int(parts["y_test"].sum()),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f"selected on validation F1: {selection['label']}")

    setup_mlflow(args.artifacts)
    for row in rows:
        with mlflow.start_run(run_name=row["model"]):
            mlflow.set_tags({"data": args.data_label, "model": row["model"]})
            mlflow.log_metrics(
                {k: float(v) for k, v in row.items() if isinstance(v, float) and np.isfinite(v)}
            )


if __name__ == "__main__":
    main()
