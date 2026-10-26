"""Read the RDD2022 country zips, split the labelled images and write a YOLO dataset.

RDD2022 stores one Pascal VOC XML file per image under
``<Country>/train/annotations/xmls/`` and the images under
``<Country>/train/images/``. The test folders hold images only, so they are
not used. Every box of every class is kept in ``annotations.parquet``; the
YOLO labels keep the four scored damage types.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from portfolio_common import sample
from portfolio_common.config import parse_args

from road_damage.config import (
    ANNOTATIONS_PATH,
    CLASS_IDS,
    CLASSES,
    COUNTRIES,
    DATA_DIR,
    EXTRACT_DIR,
    IMAGES_PATH,
    KNOWN_COUNTRIES,
    RAW_DIR,
    SEED,
    SPLIT_BLOCK,
    SPLIT_FRACTIONS,
    YOLO_DIR,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
LIGHTING_BINS = ("dark", "medium", "bright")


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def extract(source: Path, dest: Path = EXTRACT_DIR) -> list[Path]:
    """Unpack every zip in ``source`` once and return the folders to scan.

    A source folder without zips is scanned as it is (an extracted copy).
    """
    zips = sorted(source.glob("*.zip"))
    if not zips:
        return [source]
    roots = []
    for archive in zips:
        target = dest / archive.stem
        marker = target / ".complete"
        if not marker.exists():
            shutil.rmtree(target, ignore_errors=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
            marker.touch()
        roots.append(target)
    return roots


def country_of(path: Path) -> str | None:
    """The country folder a file sits in, or None when no part of the path names one."""
    for part in reversed(path.parts):
        if part in KNOWN_COUNTRIES:
            return part
    return None


def parse_xml(path: Path) -> dict:
    """One VOC annotation file as a dictionary: file name, size and boxes."""
    root = ET.parse(path).getroot()
    if root.tag != "annotation":
        raise SchemaError(f"{path}: root element is <{root.tag}>, expected <annotation>")
    size = root.find("size")
    width = float(size.findtext("width")) if size is not None else float("nan")
    height = float(size.findtext("height")) if size is not None else float("nan")
    boxes = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise SchemaError(f"{path}: an <object> has no <bndbox>")
        boxes.append(
            {
                "label": (obj.findtext("name") or "").strip(),
                **{k: float(bndbox.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")},
            }
        )
    return {
        "filename": (root.findtext("filename") or path.with_suffix(".jpg").name).strip(),
        "width": width,
        "height": height,
        "boxes": boxes,
    }


def find_image(xml_path: Path, filename: str) -> Path | None:
    images_dir = xml_path.parent.parent.parent / "images"  # <split>/annotations/xmls/x.xml -> <split>/images
    candidates = [images_dir / filename] + [images_dir / (xml_path.stem + s) for s in IMAGE_SUFFIXES]
    return next((c for c in candidates if c.exists()), None)


def brightness(path: Path) -> float:
    """Mean luminance from 0 to 255, on a thumbnail."""
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((128, 128))
        return float(np.asarray(gray, dtype=np.float32).mean())


def scan(roots: list[Path], countries: tuple[str, ...] = COUNTRIES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Image and box tables of the labelled (train) images of the chosen countries."""
    image_rows, box_rows, missing = [], [], 0
    for root in roots:
        for xml_path in sorted(root.rglob("*.xml")):
            country = country_of(xml_path)
            if country not in countries or "annotations" not in xml_path.parts:
                continue
            record = parse_xml(xml_path)
            image_path = find_image(xml_path, record["filename"])
            if image_path is None:
                missing += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            image_id = image_path.stem
            image_rows.append(
                {
                    "image_id": image_id,
                    "country": country,
                    "path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                    "xml_width": record["width"],
                    "xml_height": record["height"],
                    "brightness": brightness(image_path),
                }
            )
            box_rows += [{"image_id": image_id, "country": country, **box} for box in record["boxes"]]
    images = pd.DataFrame(image_rows)
    if images.empty:
        raise SchemaError(f"no labelled images of {', '.join(countries)} found under {roots}")
    if images["image_id"].duplicated().any():
        raise SchemaError("image file names are not unique across countries")
    found = set(images["country"])
    if found != set(countries):
        raise SchemaError(f"countries found: {sorted(found)}; expected {sorted(countries)}")
    boxes = pd.DataFrame(box_rows, columns=["image_id", "country", "label", "xmin", "ymin", "xmax", "ymax"])
    images.attrs["missing_images"] = missing
    return images, boxes


def assign_splits(images: pd.DataFrame, seed: int = SEED, block: int = SPLIT_BLOCK) -> pd.Series:
    """Train, validation or test per image, by blocks of consecutive file names within each country."""
    rng = np.random.default_rng(seed)
    names, cuts = list(SPLIT_FRACTIONS), np.cumsum(list(SPLIT_FRACTIONS.values()))
    split = pd.Series(index=images.index, dtype=object)
    for _, group in images.sort_values("image_id").groupby("country", sort=True):
        blocks = np.arange(len(group)) // block
        n_blocks = blocks.max() + 1
        order = rng.permutation(n_blocks)
        # Blocks in random order fill train, then validation, then test, to the target fractions.
        position = np.empty(n_blocks)
        position[order] = (np.arange(n_blocks) + 0.5) / n_blocks
        block_split = np.array(names)[np.searchsorted(cuts, position)]
        split.loc[group.index] = block_split[blocks]
    return split


def lighting_bins(values: pd.Series) -> pd.Series:
    """Dark, medium or bright: tertiles of image brightness over all images."""
    return pd.qcut(values, 3, labels=list(LIGHTING_BINS)).astype(str)


def yolo_lines(boxes: pd.DataFrame, width: int, height: int) -> tuple[list[str], int]:
    """Label lines ``class cx cy w h`` (normalized) and the number of boxes dropped as degenerate."""
    lines, dropped = [], 0
    for box in boxes.itertuples():
        x0, x1 = np.clip([box.xmin, box.xmax], 0, width)
        y0, y1 = np.clip([box.ymin, box.ymax], 0, height)
        if x1 - x0 < 1 or y1 - y0 < 1:
            dropped += 1
            continue
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"{CLASS_IDS[box.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, dropped


def write_yolo(images: pd.DataFrame, boxes: pd.DataFrame, out: Path = YOLO_DIR) -> dict:
    """Image links, label files, split lists and ``dataset.yaml``; returns counts for the audit."""
    shutil.rmtree(out, ignore_errors=True)
    scored = boxes[boxes["label"].isin(CLASSES)]
    by_image = dict(tuple(scored.groupby("image_id")))
    dropped = 0
    lists: dict[str, list[str]] = {}
    for row in images.itertuples():
        source = Path(row.path)
        image_link = out / "images" / row.split / source.name
        label_path = out / "labels" / row.split / f"{source.stem}.txt"
        image_link.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_link.symlink_to(source)
        lines, n = yolo_lines(by_image.get(row.image_id, scored.iloc[:0]), row.width, row.height)
        dropped += n
        label_path.write_text("".join(f"{line}\n" for line in lines))
        entry = f"./images/{row.split}/{source.name}"
        for key in (row.split, f"{row.split}_{row.country}", f"{row.split}_{row.lighting}"):
            lists.setdefault(key, []).append(entry)
    for key, entries in lists.items():
        (out / f"{key}.txt").write_text("".join(f"{e}\n" for e in sorted(entries)))
    write_dataset_yaml(out, out / "dataset.yaml", test="test.txt")
    return {"degenerate_boxes": dropped, "lists": {k: len(v) for k, v in sorted(lists.items())}}


def write_dataset_yaml(root: Path, path: Path, test: str) -> Path:
    """An Ultralytics dataset file; ``test`` names the list that ``split='test'`` evaluates."""
    spec = {
        "path": str(root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": test,
        "names": {i: code for code, i in CLASS_IDS.items()},
    }
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def sample_blocks(
    images: pd.DataFrame, boxes: pd.DataFrame, share: float, seed: int, block: int = SPLIT_BLOCK
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Keep whole blocks of consecutive images, stratified by country and split.

    The blocks are those of ``assign_splits``, so the sample keeps neighbor images apart in the
    same way as the full split. Returns the kept images, their boxes and the number of blocks
    before and after the sample.
    """
    position = images.sort_values("image_id").groupby("country").cumcount()
    key = images["country"] + "_" + (position // block).astype(str)
    keep = sample.select(key, share, seed, strata=images["country"] + "_" + images["split"])
    kept = images[keep].copy()
    kept.attrs = images.attrs
    kept_boxes = boxes[boxes["image_id"].isin(kept["image_id"])].copy()
    return kept, kept_boxes, key.nunique(), key[keep].nunique()


def build(
    source: Path,
    data_dir: Path = DATA_DIR,
    countries: tuple[str, ...] = COUNTRIES,
    block: int = SPLIT_BLOCK,
    share: float = sample.FULL,
    seed: int = sample.DEFAULT_SEED,
) -> dict:
    roots = extract(source, data_dir / EXTRACT_DIR.name)
    images, boxes = scan(roots, countries)
    images["split"] = assign_splits(images, block=block)
    images, boxes, before, after = sample_blocks(images, boxes, share, seed, block)
    sample.record(data_dir, share, seed, "block of consecutive images", before, after)
    empty = {f"{c} {s}" for c in countries for s in SPLIT_FRACTIONS} - set(
        images["country"] + " " + images["split"]
    )
    if empty:
        raise SchemaError(f"empty splits: {sorted(empty)}; use a smaller --block for a small source")
    images["lighting"] = lighting_bins(images["brightness"])
    counts = write_yolo(images, boxes, data_dir / YOLO_DIR.name)
    counts["missing_images"] = images.attrs.get("missing_images", 0)
    images.to_parquet(data_dir / IMAGES_PATH.name, index=False)
    boxes.to_parquet(data_dir / ANNOTATIONS_PATH.name, index=False)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=RAW_DIR, help="folder with RDD2022_<Country>.zip files"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--countries", nargs="+", default=list(COUNTRIES))
    parser.add_argument("--block", type=int, default=SPLIT_BLOCK, help="consecutive images kept in one split")
    sample.add_arguments(parser, "blocks of consecutive images, stratified by country and split")
    args = parse_args(parser, __file__)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    counts = build(
        args.source, args.data_dir, tuple(args.countries), args.block, args.sample, args.sample_seed
    )
    print(yaml.safe_dump(counts, sort_keys=False))


if __name__ == "__main__":
    main()
