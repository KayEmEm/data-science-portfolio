"""Export the main model to ONNX and measure inference speed on CPU.

Speed is the median time per image at batch size 1 over the first test
images, after a warm up, for the PyTorch weights and for the ONNX model in
ONNX Runtime. The ONNX model is also scored on the test split, to show that
the export keeps the accuracy.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from portfolio_common.config import parse_args
from ultralytics import YOLO, settings

from road_damage.config import ARTIFACTS_DIR, IMAGE_SIZE, YOLO_DIR
from road_damage.evaluate import scores, validate
from road_damage.train import MAIN_RUN, weights_path


def test_images(yolo_dir: Path, n: int) -> list[str]:
    entries = (yolo_dir / "test.txt").read_text().split()
    return [str(yolo_dir / e) for e in entries[:n]]


def latency_ms(model: YOLO, images: list[str], imgsz: int, warmup: int = 5) -> dict:
    """Median and 90th percentile milliseconds per image, preprocessing and postprocessing included."""
    for path in images[:warmup]:
        model.predict(path, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for path in images:
        started = time.perf_counter()
        model.predict(path, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
        "images": len(times),
    }


def export(yolo_dir: Path, artifacts_dir: Path, imgsz: int, n_images: int) -> dict:
    weights = weights_path(artifacts_dir, MAIN_RUN)
    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True))
    target = artifacts_dir / "road_damage_yolo11n.onnx"
    shutil.copy(onnx_path, target)
    images = test_images(yolo_dir, n_images)
    torch_model, onnx_model = YOLO(str(weights)), YOLO(str(target), task="detect")
    onnx_scores = validate(
        target, yolo_dir / "dataset.yaml", artifacts_dir / "eval", "onnx", imgsz, 1, plots=False
    )
    result = {
        "cpu": {"threads": torch.get_num_threads(), "cores": os.cpu_count()},
        "versions": {"torch": torch.__version__, "onnxruntime": onnxruntime.__version__},
        "imgsz": imgsz,
        "onnx_megabytes": round(target.stat().st_size / 2**20, 1),
        "pytorch": latency_ms(torch_model, images, imgsz),
        "onnxruntime": latency_ms(onnx_model, images, imgsz),
        "onnx_test": {k: v for k, v in scores(onnx_scores).items() if k != "ap50_per_class"},
    }
    (artifacts_dir / "export_results.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yolo-dir", type=Path, default=YOLO_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--images", type=int, default=200, help="test images timed per runtime")
    args = parse_args(parser, __file__)
    settings.update({"mlflow": False})
    print(json.dumps(export(args.yolo_dir, args.artifacts, args.imgsz, args.images), indent=2))


if __name__ == "__main__":
    main()
