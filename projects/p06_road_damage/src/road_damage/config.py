"""Paths, classes, split and training settings for project 6."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # country zips of RDD2022, as supplied
EXTRACT_DIR = DATA_DIR / "extracted"
YOLO_DIR = DATA_DIR / "yolo"  # images (symbolic links), labels and split lists in the Ultralytics layout
ANNOTATIONS_PATH = DATA_DIR / "annotations.parquet"  # one row per box, all classes
IMAGES_PATH = DATA_DIR / "images.parquet"  # one row per image: country, split, size, brightness
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

# The official S3 links of RDD2022 answer AccessDenied and figshare does not
# resolve from the build environment, so the country zips are supplied by hand
# (``make data SOURCE=<folder>``). See the README for the steps.
COUNTRIES = ("Czech", "India")
KNOWN_COUNTRIES = ("Czech", "India", "Japan", "Norway", "United_States", "China_MotorBike", "China_Drone")

# The four damage types that CRDDC2022 scores. Other labels (for example D43,
# D44, D50 and Repair in the India data) are counted in the audit and dropped.
CLASSES = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}
CLASS_IDS = {code: i for i, code in enumerate(CLASSES)}

# The RDD2022 test images carry no labels, so the labelled train images are
# split into train, validation and test. Images are taken in sequence along a
# road and neighbours look alike, so the split assigns blocks of consecutive
# images (by file name) rather than single images.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_BLOCK = 25
SEED = 42

# Training on 4 CPU cores. YOLO11n has 2.6 million parameters; its COCO
# weights come from the Ultralytics GitHub release (AGPL-3.0).
BASE_WEIGHTS = "yolo11n.pt"
IMAGE_SIZE = 640
BATCH = 16
EPOCHS = 30
PATIENCE = 10
WORKERS = 4
EXPERIMENT = "p06_road_damage"
