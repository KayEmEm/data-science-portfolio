import zipfile

import numpy as np
import pandas as pd
import pytest

from illicit_graph.config import CLASSES_FILE, EDGES_FILE, FEATURES_FILE, RAW_FILES, SPLIT_STEPS
from illicit_graph.data import SchemaError, build, extract, prepare, read_classes, read_features


def test_prepare_keeps_every_transaction_and_edge(release_folder, prepared):
    nodes, edges, report = prepared
    raw_edges = pd.read_csv(release_folder / EDGES_FILE)
    assert len(nodes) == 6_000
    assert report["counts_found"]["edges"] == len(raw_edges)
    assert len(edges) == len(raw_edges.drop_duplicates())
    assert report["edges_across_time_steps"] == 0
    assert report["nodes_without_class_row"] == 0
    # edges point at row positions whose ids are the ones of the edge list
    assert (nodes["tx_id"].to_numpy()[edges["src"]] == edges["src_id"].to_numpy()).all()


def test_split_follows_time_steps(prepared):
    nodes = prepared[0]
    for name, steps in SPLIT_STEPS.items():
        assert set(nodes.loc[nodes["split"] == name, "time_step"]) <= set(steps)
    assert nodes["split"].notna().all()
    assert set(nodes["label"].unique()) <= {-1, 0, 1}


def test_zip_with_subfolder_is_extracted(release_folder, tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in RAW_FILES:
            handle.write(release_folder / name, f"elliptic_bitcoin_dataset/{name}")
    target = extract(archive, tmp_path / "raw")
    assert sorted(p.name for p in target.iterdir()) == sorted(RAW_FILES)


def test_missing_file_is_reported(release_folder, tmp_path):
    (tmp_path / FEATURES_FILE).write_bytes((release_folder / FEATURES_FILE).read_bytes())
    with pytest.raises(FileNotFoundError, match="lacks"):
        prepare(tmp_path)


def test_wrong_column_count_is_rejected(tmp_path):
    path = tmp_path / FEATURES_FILE
    pd.DataFrame(np.ones((3, 10))).to_csv(path, header=False, index=False)
    with pytest.raises(SchemaError, match="columns"):
        read_features(path)


def test_unknown_class_value_is_rejected(tmp_path):
    path = tmp_path / CLASSES_FILE
    pd.DataFrame({"txId": [1, 2], "class": ["1", "3"]}).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="class values"):
        read_classes(path)


def test_self_loops_duplicates_and_unknown_ids_are_dropped(prepared):
    nodes = prepared[0]
    ids = nodes["tx_id"].to_numpy()
    features = nodes.drop(columns=["split", "label"])
    classes = nodes[["tx_id", "label"]]
    edges = pd.DataFrame(
        {"src_id": [ids[0], ids[0], ids[1], ids[2]], "dst_id": [ids[1], ids[1], ids[1], 999]}
    )
    _, out, issues = build(features, classes, edges)
    assert len(out) == 1
    assert issues["duplicate_edges"] == 1
    assert issues["self_loops"] == 1
    assert issues["edges_with_unknown_node"] == 1
