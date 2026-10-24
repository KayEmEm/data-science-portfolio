import numpy as np
import pandas as pd
import pytest

from flow_anomaly.config import FULL_FILES, NORMAL, NUMERIC
from flow_anomaly.data import SchemaError, detect_layout, prepare, stratified_split


def test_full_layout_is_read_with_canonical_names(full_folder, raw_flows):
    flows, audit = prepare(full_folder)
    assert audit["layout"] == "full"
    assert len(flows) == len(raw_flows)
    assert set(NUMERIC) <= set(flows.columns)
    # " Fuzzers " and "Backdoors" in the files, a blank cell for normal flows
    assert {"Fuzzers", "Backdoor", NORMAL} <= set(flows["attack_cat"])
    assert (flows.loc[flows["label"] == 0, "attack_cat"] == NORMAL).all()
    assert audit["label_0_with_attack_type"] == 0
    assert audit["label_1_without_attack_type"] == 0
    assert not audit["counts_match_documented"]


def test_full_layout_without_features_file_uses_builtin_names(full_folder, tmp_path):
    for name in FULL_FILES:
        (tmp_path / name).write_bytes((full_folder / name).read_bytes())
    flows, _ = prepare(tmp_path)
    assert flows["label"].sum() > 0


def test_partition_layout_keeps_its_test_file(partition_folder):
    flows, audit = prepare(partition_folder)
    assert audit["layout"] == "partition"
    assert (flows.loc[flows["source_role"] == "test", "split"] == "test").all()
    assert set(flows.loc[flows["source_role"] == "train", "split"]) == {"train", "validation"}


def test_wrong_column_count_is_refused(full_folder, tmp_path):
    for name in FULL_FILES:
        frame = pd.read_csv(full_folder / name, header=None)
        frame.iloc[:, :-1].to_csv(tmp_path / name, header=False, index=False)
    with pytest.raises(SchemaError):
        prepare(tmp_path)


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_layout(tmp_path)


def test_non_numeric_cells_are_counted(full_folder, tmp_path):
    for name in FULL_FILES:
        frame = pd.read_csv(full_folder / name, header=None)
        frame[6] = frame[6].astype(object)
        frame.iloc[0, 6] = "-"  # dur
        frame.to_csv(tmp_path / name, header=False, index=False)
    flows, audit = prepare(tmp_path)
    assert audit["not_numeric_cells"] == {"dur": len(FULL_FILES)}
    assert np.isfinite(flows["dur"]).all()


def test_split_keeps_shares_within_each_type():
    strata = pd.Series(["a"] * 1000 + ["b"] * 50)
    split = stratified_split(strata, {"train": 0.6, "validation": 0.2, "test": 0.2}, seed=0)
    for value, n in (("a", 1000), ("b", 50)):
        part = split[strata.to_numpy() == value]
        assert (part == "train").sum() == round(0.6 * n)
        assert (part == "test").sum() == n - round(0.8 * n)
