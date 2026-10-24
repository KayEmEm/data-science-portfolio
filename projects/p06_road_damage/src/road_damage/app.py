"""Gradio demo: upload a street image and see the detected road damage.

Run with ``make demo`` after ``make train`` (or with any weights file:
``python -m road_damage.app --weights <file>``). It serves on
http://127.0.0.1:7860 and sends no data elsewhere.
"""

import argparse
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

from road_damage.config import ARTIFACTS_DIR, CLASSES, IMAGE_SIZE

DEFAULT_WEIGHTS = ARTIFACTS_DIR / "road_damage_yolo11n.onnx"


def build_demo(weights: Path, imgsz: int = IMAGE_SIZE) -> gr.Interface:
    model = YOLO(str(weights), task="detect")

    def detect(image, confidence: float):
        result = model.predict(image, imgsz=imgsz, conf=confidence, device="cpu", verbose=False)[0]
        rows = [
            [CLASSES[result.names[int(c)]], round(float(p), 3), *[round(float(v)) for v in xyxy]]
            for c, p, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy, strict=True)
        ]
        return result.plot()[:, :, ::-1], rows  # plot() returns BGR

    return gr.Interface(
        fn=detect,
        inputs=[gr.Image(type="numpy", label="Street image"), gr.Slider(0.05, 0.9, 0.25, label="Confidence")],
        outputs=[
            gr.Image(label="Detections"),
            gr.Dataframe(headers=["damage type", "confidence", "x0", "y0", "x1", "y1"], label="Boxes"),
        ],
        title="Road damage detection (YOLO11n, RDD2022 Czech and India)",
        description="Cracks (longitudinal, transverse, alligator) and potholes. "
        "Research demo, not for operational use.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    build_demo(args.weights, args.imgsz).launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
