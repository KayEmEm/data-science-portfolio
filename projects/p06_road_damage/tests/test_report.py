import pandas as pd

from road_damage.report import audit


def test_audit_counts_scored_and_dropped_boxes(data_dir):
    images = pd.read_parquet(data_dir / "images.parquet")
    boxes = pd.read_parquet(data_dir / "annotations.parquet")
    text, numbers = audit(images, boxes)
    assert numbers["images"] == 80
    assert numbers["boxes_dropped_other_labels"] == 2  # one D43 box per country
    assert numbers["boxes_scored"] == int(boxes["label"].isin(["D00", "D10", "D20", "D40"]).sum())
    assert "not scored, dropped" in text
