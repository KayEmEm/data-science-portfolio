"""Fine tune YOLO11n on the RDD2022 split, one run per training configuration.

Three runs share the data, image size, seed and number of epochs:

* ``frozen_backbone``: the baseline. The COCO backbone stays fixed and only the
  neck and detection head learn the four damage types.
* ``full``: every layer learns, with the default Ultralytics augmentation
  (mosaic, scale, translation, horizontal flip, color jitter). The main model.
* ``light_augment``: as ``full`` but without mosaic and with smaller scale and
  color changes, for the augmentation study.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings
from ultralytics.utils.downloads import attempt_download_asset

from road_damage.config import (
    ARTIFACTS_DIR,
    BASE_WEIGHTS,
    BATCH,
    EPOCHS,
    EXPERIMENT,
    IMAGE_SIZE,
    PATIENCE,
    SEED,
    WORKERS,
    YOLO_DIR,
)

RUNS: dict[str, dict] = {
    "frozen_backbone": {"freeze": 11},  # layers 0 to 10 of YOLO11n form the backbone
    "full": {},
    "light_augment": {"mosaic": 0.0, "scale": 0.2, "hsv_s": 0.3, "hsv_v": 0.2, "translate": 0.05},
}
MAIN_RUN = "full"


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def resolve_weights(weights: str, artifacts_dir: Path) -> str:
    """A local path for released weights (downloaded once from the Ultralytics GitHub release).

    A model file such as ``yolo11n.yaml`` is passed through: it builds the
    network with random weights, which the smoke run uses to avoid a download.
    """
    if not weights.endswith(".pt") or Path(weights).exists():
        return weights
    target = artifacts_dir / "weights" / Path(weights).name
    target.parent.mkdir(parents=True, exist_ok=True)
    return attempt_download_asset(target)


def weights_path(artifacts_dir: Path, run: str) -> Path:
    return artifacts_dir / "runs" / run / "weights" / "best.pt"


def train_run(
    run: str,
    data_yaml: Path,
    artifacts_dir: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
) -> dict:
    """Train one configuration and return its settings, time and last validation scores."""
    overrides = RUNS[run]
    started = time.perf_counter()
    model = YOLO(resolve_weights(weights, artifacts_dir))
    result = model.train(
        data=str(data_yaml),
        project=str((artifacts_dir / "runs").resolve()),
        name=run,
        exist_ok=True,
        epochs=epochs,
        patience=PATIENCE,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device="cpu",
        seed=SEED,
        deterministic=True,
        plots=True,
        verbose=False,
        **overrides,
    )
    minutes = (time.perf_counter() - started) / 60
    summary = {
        "run": run,
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "overrides": overrides,
        "train_minutes": round(minutes, 1),
        "val_map50": float(result.box.map50),
        "val_map50_95": float(result.box.map),
    }
    with mlflow.start_run(run_name=run):
        mlflow.set_tags({"stage": "train", "run": run})
        mlflow.log_params({k: v for k, v in summary.items() if k not in ("run", "overrides")} | overrides)
        mlflow.log_metrics({k: summary[k] for k in ("train_minutes", "val_map50", "val_map50_95")})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=YOLO_DIR / "dataset.yaml")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument(
        "--weights", default=BASE_WEIGHTS, help="COCO weights, or yolo11n.yaml to start untrained"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    settings.update({"mlflow": False})  # runs are logged once, by this module, to the project's own store
    setup_mlflow(args.artifacts)
    summary_path = args.artifacts / "train_summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for run in args.runs:
        summaries[run] = train_run(
            run, args.data, args.artifacts, args.weights, args.epochs, args.imgsz, args.batch, args.workers
        )
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summaries[run]))


if __name__ == "__main__":
    main()
