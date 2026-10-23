import gzip

import pandas as pd
import pytest

from email_uplift import data
from email_uplift.config import REQUIRED_COLUMNS
from email_uplift.data import TREATMENT_COL, SchemaError, feature_matrix, normalize
from email_uplift.synthetic import make_customers


@pytest.fixture
def raw():
    return make_customers(300)


def test_normalize_adds_treatment_codes(raw):
    out = normalize(raw, expected_rows=None)
    codes = out.groupby("segment")[TREATMENT_COL].unique().map(lambda v: v.tolist()).to_dict()
    assert codes == {"No E-Mail": [0], "Mens E-Mail": [1], "Womens E-Mail": [2]}
    assert list(out.columns) == [*REQUIRED_COLUMNS, TREATMENT_COL]


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda f: f.drop(columns=["spend"]), "missing columns"),
        (lambda f: f.assign(segment=f["segment"].where(f.index > 0, "Kids E-Mail")), "unknown values"),
        (lambda f: f.assign(visit=f["visit"] * 2), "not binary"),
        (lambda f: f.assign(recency=f["recency"] + 12), "recency"),
        (lambda f: f.assign(spend=f["spend"].where(f["conversion"] == 1, 5.0)), "without a conversion"),
    ],
)
def test_normalize_rejects_bad_tables(raw, corrupt, message):
    with pytest.raises(SchemaError, match=message):
        normalize(corrupt(raw), expected_rows=None)


def test_normalize_checks_row_count(raw):
    with pytest.raises(SchemaError, match="expected 64000 rows"):
        normalize(raw)


def test_feature_matrix_is_numeric_and_pre_treatment(raw):
    features = feature_matrix(normalize(raw, expected_rows=None))
    assert features.dtypes.map(lambda d: d.kind == "f").all()
    assert not {"visit", "conversion", "spend", "segment", TREATMENT_COL} & set(features.columns)
    assert {"zip_surburban", "zip_rural", "channel_web", "channel_multichannel"} <= set(features.columns)


def test_download_rejects_a_file_with_the_wrong_checksum(tmp_path):
    target = tmp_path / data.RAW_NAME
    with gzip.open(target, "wt") as handle:
        handle.write("not,the,file\n")
    with pytest.raises(SchemaError, match="SHA256"):
        data.download(tmp_path)


def test_main_reads_a_local_copy(tmp_path, raw, monkeypatch):
    source = tmp_path / "copy.csv.gz"
    raw.to_csv(source, index=False)
    out = tmp_path / "customers.parquet"
    argv = ["data", "--source", str(source), "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SchemaError, match="expected 64000 rows"):
        data.main()
    monkeypatch.setattr("sys.argv", [*argv, "--expected-rows", str(len(raw))])
    data.main()
    pd.testing.assert_frame_equal(pd.read_parquet(out), normalize(raw, expected_rows=None))
