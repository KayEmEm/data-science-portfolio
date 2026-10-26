"""Synthetic sources for the tests and ``make smoke``, made by the synthetic modules of projects 1, 6
and 8, and an untrained detector in the layout of the project 6 weights.

The numbers they give say nothing about real credit risk, network traffic or roads.
"""

import argparse
from pathlib import Path

import torch
from flow_anomaly.synthetic import make_flows, write_partition
from loan_default.synthetic import make_source
from portfolio_common.config import parse_args
from road_damage.config import CLASS_IDS
from road_damage.synthetic import write_archives

from adv_robustness.config import IMAGE_COUNTRIES, SEED


def untrained_detector(path: Path, image_size: int) -> Path:
    """YOLO11n with random weights and the four damage classes, saved as Ultralytics weights."""
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(SEED)
    model = DetectionModel("yolo11n.yaml", nc=len(CLASS_IDS), verbose=False)
    model.names = {i: code for code, i in CLASS_IDS.items()}
    model.args = {"imgsz": image_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.half(), "train_args": model.args}, path)
    return path


def write(
    out: Path, clients: int = 1500, flows: int = 12_000, images: int = 40, image_size: int = 160
) -> None:
    (out / "credit").mkdir(parents=True, exist_ok=True)
    make_source(clients, SEED).to_parquet(out / "credit" / "clients.parquet", index=False)
    write_partition(out / "flows", make_flows(flows, seed=SEED))
    write_archives(out / "images", images, countries=IMAGE_COUNTRIES)
    untrained_detector(out / "detector" / "untrained.pt", image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--clients", type=int, default=1500)
    parser.add_argument("--flows", type=int, default=12_000)
    parser.add_argument("--images", type=int, default=40, help="labelled images per country")
    args = parse_args(parser, __file__)
    write(args.out, args.clients, args.flows, args.images)


if __name__ == "__main__":
    main()
