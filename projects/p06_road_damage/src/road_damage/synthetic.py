"""Small synthetic country zips in the RDD2022 layout, for the tests and ``make smoke``.

The images are gray noise with dark rectangles where the boxes are; they say
nothing about real roads. Each country has one image without damage, one box
of a label outside the four scored classes and one box that crosses the image
border, as the real data does.
"""

import argparse
import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from road_damage.config import CLASSES, COUNTRIES, SEED

SIZE = 160


def annotation_xml(filename: str, boxes: list[tuple[str, float, float, float, float]]) -> bytes:
    """A Pascal VOC annotation in the layout of the RDD2022 files."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    for key, value in (("width", SIZE), ("height", SIZE), ("depth", 3)):
        ET.SubElement(size, key).text = str(value)
    ET.SubElement(root, "segmented").text = "0"
    for label, x0, y0, x1, y1 in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        for key in ("pose", "truncated", "difficult"):
            ET.SubElement(obj, key).text = "Unspecified" if key == "pose" else "0"
        bndbox = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x0, y0, x1, y1), strict=True):
            ET.SubElement(bndbox, key).text = f"{value:.1f}"
    return ET.tostring(root, encoding="utf-8")


def make_image(rng: np.random.Generator, boxes, level: float) -> bytes:
    pixels = rng.normal(level, 12, size=(SIZE, SIZE, 3))
    for _, x0, y0, x1, y1 in boxes:
        pixels[int(max(y0, 0)) : int(min(y1, SIZE)), int(max(x0, 0)) : int(min(x1, SIZE))] = level - 70
    buffer = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write_country(path: Path, country: str, n_images: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    labels = list(CLASSES)
    with zipfile.ZipFile(path, "w") as archive:
        for k in range(1, n_images + 1):
            name = f"{country}_{k:06d}"
            boxes = []
            if k > 1:  # the first image has no damage
                for _ in range(rng.integers(1, 3)):
                    w, h = rng.uniform(20, 60, size=2)
                    x0, y0 = rng.uniform(0, SIZE - w), rng.uniform(0, SIZE - h)
                    boxes.append(
                        (labels[rng.integers(len(labels))], round(x0, 1), round(y0, 1), x0 + w, y0 + h)
                    )
            if k == 2:
                boxes.append(("D43", 10, 10, 30, 30))
            if k == 3:
                boxes.append(("D00", SIZE - 20, 50, SIZE + 15, 90))
            level = rng.uniform(90, 200)
            archive.writestr(f"{country}/train/images/{name}.jpg", make_image(rng, boxes, level))
            archive.writestr(
                f"{country}/train/annotations/xmls/{name}.xml", annotation_xml(f"{name}.jpg", boxes)
            )
        archive.writestr(f"{country}/test/images/{country}_900001.jpg", make_image(rng, [], 128))
    return path


def write_archives(out_dir: Path, n_images: int = 40, countries=COUNTRIES) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, country in enumerate(countries):
        write_country(out_dir / f"RDD2022_{country}.zip", country, n_images, SEED + i)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--images", type=int, default=40, help="labelled images per country")
    args = parser.parse_args()
    write_archives(args.out_dir, args.images)


if __name__ == "__main__":
    main()
