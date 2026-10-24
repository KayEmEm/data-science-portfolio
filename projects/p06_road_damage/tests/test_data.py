import numpy as np
import pandas as pd
import pytest
import yaml

from road_damage.config import CLASS_IDS, SPLIT_BLOCK
from road_damage.data import SchemaError, assign_splits, country_of, extract, parse_xml, scan, yolo_lines
from road_damage.synthetic import annotation_xml


def test_parse_xml_reads_float_coordinates(tmp_path):
    path = tmp_path / "Czech_000001.xml"
    path.write_bytes(annotation_xml("Czech_000001.jpg", [("D40", 1.5, 2.0, 30.25, 40.0)]))
    record = parse_xml(path)
    assert record["filename"] == "Czech_000001.jpg"
    assert record["boxes"] == [{"label": "D40", "xmin": 1.5, "ymin": 2.0, "xmax": 30.2, "ymax": 40.0}]


def test_parse_xml_rejects_other_root(tmp_path):
    path = tmp_path / "x.xml"
    path.write_text("<svg/>")
    with pytest.raises(SchemaError):
        parse_xml(path)


def test_country_of_reads_the_folder_name(tmp_path):
    assert country_of(tmp_path / "RDD2022" / "India" / "train" / "images" / "a.jpg") == "India"
    assert country_of(tmp_path / "elsewhere" / "a.jpg") is None


def test_scan_keeps_labelled_images_only(source, tmp_path):
    images, boxes = scan(extract(source, tmp_path))
    assert images.groupby("country").size().to_dict() == {"Czech": 40, "India": 40}
    assert not images["image_id"].str.contains("900001").any()  # test folders carry no labels
    assert "D43" in set(boxes["label"])


def test_scan_rejects_a_missing_country(source, tmp_path):
    with pytest.raises(SchemaError):
        scan(extract(source, tmp_path), countries=("Czech", "Japan"))


def test_splits_keep_blocks_together_and_hit_fractions():
    images = pd.DataFrame({"image_id": [f"Czech_{i:06d}" for i in range(1000)], "country": "Czech"})
    split = assign_splits(images)
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)
    blocks = images.index // SPLIT_BLOCK
    assert split.groupby(blocks).nunique().eq(1).all()
    assert split.equals(assign_splits(images))  # seeded


def test_yolo_lines_clip_and_normalize():
    boxes = pd.DataFrame(
        {
            "label": ["D00", "D40", "D10"],
            "xmin": [0, 90, 50],
            "ymin": [0, 10, 50],
            "xmax": [50, 120, 50.5],
            "ymax": [100, 30, 60],
        }
    )
    lines, dropped = yolo_lines(boxes, 100, 100)
    assert dropped == 1  # narrower than one pixel
    first = np.array(lines[0].split(), dtype=float)
    assert first.tolist() == [CLASS_IDS["D00"], 0.25, 0.5, 0.5, 1.0]
    second = np.array(lines[1].split(), dtype=float)
    assert second[1] + second[3] / 2 == pytest.approx(1.0)  # clipped at the right border


def test_build_writes_an_ultralytics_dataset(data_dir):
    yolo = data_dir / "yolo"
    spec = yaml.safe_load((yolo / "dataset.yaml").read_text())
    assert spec["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    entries = {s: (yolo / f"{s}.txt").read_text().split() for s in ("train", "val", "test")}
    assert sum(map(len, entries.values())) == 80
    assert not set(entries["train"]) & set(entries["test"])
    for entry in entries["test"]:
        assert (yolo / entry).resolve().exists()
        label = yolo / entry.replace("./images/", "labels/", 1)
        for line in label.with_suffix(".txt").read_text().splitlines():
            assert int(line.split()[0]) in CLASS_IDS.values()
    subsets = {p.stem.removeprefix("test_") for p in yolo.glob("test_*.txt")}
    assert {"Czech", "India"} <= subsets <= {"Czech", "India", "dark", "medium", "bright"}
    images = pd.read_parquet(data_dir / "images.parquet")
    assert set(images["lighting"]) == {"dark", "medium", "bright"}
