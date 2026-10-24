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
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
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


def backup_callback(backup_dir: Path):
    """An Ultralytics callback that copies the run's checkpoints and history after each epoch.

    The container that trains can be stopped without notice; the copies let a
    later ``make train`` resume from the last finished epoch.
    """

    def copy(trainer) -> None:
        save_dir = Path(trainer.save_dir)
        target = backup_dir / save_dir.name
        (target / "weights").mkdir(parents=True, exist_ok=True)
        for name in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
            if (save_dir / name).exists():
                partial = target / f"{name}.partial"
                shutil.copy2(save_dir / name, partial)
                partial.replace(target / name)  # a stop during the copy never leaves a truncated file

    return copy


def restore(run_dir: Path, backup_dir: Path | None) -> None:
    """Copy a backed up checkpoint into place when the local one is missing or older."""
    if backup_dir is None:
        return
    source = backup_dir / run_dir.name
    for name in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
        src, dst = source / name, run_dir / name
        if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def train_minutes(run_dir: Path) -> float:
    """Training time over all sessions of a run, in minutes.

    The ``time`` column of results.csv counts seconds from the start of each
    session and restarts on resume, so the last value of every session is summed.
    """
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    seconds = history["time"].to_numpy(dtype=float)
    session_ends = np.append(seconds[1:] < seconds[:-1], True)
    return round(float(seconds[session_ends].sum()) / 60, 1)


def train_run(
    run: str,
    data_yaml: Path,
    artifacts_dir: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    backup_dir: Path | None = None,
) -> dict:
    """Train one configuration, or resume it from its last epoch, and return its settings and scores."""
    overrides = RUNS[run]
    run_dir = artifacts_dir / "runs" / run
    restore(run_dir, backup_dir)
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        model = YOLO(str(last))
        if backup_dir is not None:
            model.add_callback("on_fit_epoch_end", backup_callback(backup_dir))
        result = model.train(resume=True)
    else:
        model = YOLO(resolve_weights(weights, artifacts_dir))
        if backup_dir is not None:
            model.add_callback("on_fit_epoch_end", backup_callback(backup_dir))
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
    summary = {
        "run": run,
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "overrides": overrides,
        "train_minutes": train_minutes(run_dir),
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
    parser.add_argument(
        "--backup", type=Path, default=None, help="folder for a copy of each run after each epoch"
    )
    args = parser.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    settings.update({"mlflow": False})  # runs are logged once, by this module, to the project's own store
    setup_mlflow(args.artifacts)
    summary_path = args.artifacts / "train_summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for run in args.runs:
        if run in summaries:
            print(f"skipped {run}: finished earlier (see {summary_path})")
            continue
        summaries[run] = train_run(
            run,
            args.data,
            args.artifacts,
            args.weights,
            args.epochs,
            args.imgsz,
            args.batch,
            args.workers,
            args.backup,
        )
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summaries[run]))


if __name__ == "__main__":
    main()
