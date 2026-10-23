import pandas as pd
import pytest

from loan_default.data import SchemaError, validate


def test_one_row_per_applicant(features):
    assert features["SK_ID_CURR"].is_unique
    assert len(features) == 1500


def test_employment_placeholder_becomes_null(features):
    assert (features["DAYS_EMPLOYED"] == 365243).sum() == 0
    flagged = features["APP_NOT_EMPLOYED_FLAG"] == 1
    assert flagged.any()
    assert features.loc[flagged, "DAYS_EMPLOYED"].isna().all()


def test_aggregates_present(features):
    for col in ["BUR_COUNT", "PREV_COUNT", "POS_MONTHS", "INST_LATE_SHARE", "CC_UTILIZATION_MEAN"]:
        assert col in features.columns


def test_ext_source_mean_ignores_missing(features):
    expected = features[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    pd.testing.assert_series_equal(features["APP_EXT_SOURCE_MEAN"], expected, check_names=False, atol=1e-9)


def test_validate_reports_missing_column(parquet_dir, tmp_path):
    for src in parquet_dir.glob("*.parquet"):
        frame = pd.read_parquet(src)
        if src.stem == "bureau":
            frame = frame.drop(columns="CREDIT_ACTIVE")
        frame.to_parquet(tmp_path / src.name, index=False)
    with pytest.raises(SchemaError, match="CREDIT_ACTIVE"):
        validate(tmp_path, ".parquet")
