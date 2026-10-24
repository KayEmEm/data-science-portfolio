"""Score every trained run on the held out test split, overall, per class, per country and per lighting.

mAP at IoU 0.5 is the plan's headline metric; mAP from IoU 0.5 to 0.95 and
precision and recall at the confidence that maximizes F1 are reported beside
it. The breakdowns use the main run only.

mAP is computed from every detection down to confidence 0.001 (the Ultralytics
default). The confusion matrix uses ``CONFUSION_CONF`` instead: at 0.001 almost
every true box finds a match among thousands of weak detections, so the matrix
would not show the errors of the model as it is used.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from ultralytics import YOLO, settings

from road_damage.config import ARTIFACTS_DIR, BATCH, CLASSES, IMAGE_SIZE, YOLO_DIR
from road_damage.data import write_dataset_yaml
from road_damage.train import MAIN_RUN, RUNS, setup_mlflow, weights_path

CONFUSION_CONF = 0.25  # the default confidence of Ultralytics predictions and of the demo


def scores(metrics) -> dict:
    """Overall and per class numbers from an Ultralytics ``DetMetrics``."""
    box = metrics.box
    out = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    per_class = dict.fromkeys(CLASSES, float("nan"))
    for i, class_id in enumerate(box.ap_class_index):
        per_class[list(CLASSES)[int(class_id)]] = float(box.ap50[i])
    out["ap50_per_class"] = per_class
    return out


def count_instances(list_path: Path) -> tuple[int, dict[str, int]]:
    """Images in a split list and boxes per class in their label files."""
    entries = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    counts = dict.fromkeys(CLASSES, 0)
    codes = list(CLASSES)
    for entry in entries:
        label = list_path.parent / entry.replace("./images/", "labels/", 1)
        for line in label.with_suffix(".txt").read_text().splitlines():
            counts[codes[int(line.split()[0])]] += 1
    return len(entries), counts


def validate(
    weights: Path,
    data_yaml: Path,
    out: Path,
    name: str,
    imgsz: int,
    batch: int,
    plots: bool,
    conf: float | None = None,
):
    """Ultralytics validation on the ``test`` list of ``data_yaml``; ``conf`` None keeps the default."""
    model = YOLO(str(weights))
    extra = {} if conf is None else {"conf": conf}
    return model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        project=str(out.resolve()),
        name=name,
        exist_ok=True,
        plots=plots,
        verbose=False,
        **extra,
    )


def subsets(yolo_dir: Path) -> dict[str, str]:
    """Test lists for the breakdowns, keyed by a readable name."""
    keys = {}
    for path in sorted(yolo_dir.glob("test_*.txt")):
        keys[path.stem.removeprefix("test_")] = path.name
    return keys


def evaluate(yolo_dir: Path, artifacts_dir: Path, imgsz: int, batch: int, runs: list[str]) -> dict:
    data_yaml = yolo_dir / "dataset.yaml"
    out = artifacts_dir / "eval"
    results: dict = {"runs": {}, "breakdown": {}, "test_counts": {}}
    n_images, per_class = count_instances(yolo_dir / "test.txt")
    results["test_counts"]["all"] = {"images": n_images, "boxes": per_class}
    for run in runs:
        weights = weights_path(artifacts_dir, run)
        if not weights.exists():
            print(f"skipped {run}: no weights at {weights}")
            continue
        metrics = validate(weights, data_yaml, out, run, imgsz, batch, plots=False)
        results["runs"][run] = scores(metrics)
        if run == MAIN_RUN:
            name = f"{MAIN_RUN}_conf{round(CONFUSION_CONF * 100)}"
            at_conf = validate(weights, data_yaml, out, name, imgsz, batch, plots=True, conf=CONFUSION_CONF)
            # Rows: predicted class, columns: true class; the last index is background.
            matrix = at_conf.confusion_matrix.matrix
            results["confusion_matrix"] = np.asarray(matrix).astype(int).tolist()
            results["confusion_conf"] = CONFUSION_CONF

    main_weights = weights_path(artifacts_dir, MAIN_RUN)
    if main_weights.exists():
        for key, list_name in subsets(yolo_dir).items():
            subset_yaml = write_dataset_yaml(yolo_dir, out / f"dataset_test_{key}.yaml", test=list_name)
            metrics = validate(main_weights, subset_yaml, out, f"{MAIN_RUN}_{key}", imgsz, batch, plots=False)
            results["breakdown"][key] = scores(metrics)
            n_images, per_class = count_instances(yolo_dir / list_name)
            results["test_counts"][key] = {"images": n_images, "boxes": per_class}
        plot = out / f"{MAIN_RUN}_conf{round(CONFUSION_CONF * 100)}" / "confusion_matrix_normalized.png"
        if plot.exists():
            shutil.copy(plot, artifacts_dir / "confusion_matrix_normalized.png")

    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tags({"stage": "evaluate"})
        for run, row in results["runs"].items():
            mlflow.log_metrics({f"{run}.test_{k}": v for k, v in row.items() if isinstance(v, float)})
    (artifacts_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    return results


def table(results: dict) -> pd.DataFrame:
    rows = []
    for run, row in results["runs"].items():
        rows.append(
            {"run": run, **{k: v for k, v in row.items() if k != "ap50_per_class"}, **row["ap50_per_class"]}
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yolo-dir", type=Path, default=YOLO_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()
    settings.update({"mlflow": False})
    setup_mlflow(args.artifacts)
    results = evaluate(args.yolo_dir, args.artifacts, args.imgsz, args.batch, args.runs)
    print(table(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
