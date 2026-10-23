import numpy as np
import pandas as pd
import pytest

from loan_default.config import SOURCE_TARGET
from loan_default.data import SchemaError, normalize, read_spreadsheet
from loan_default.synthetic import make_source


def test_one_row_per_client(features):
    assert features["CLIENT_ID"].is_unique
    assert len(features) == 2000


def test_categoricals_are_text(features):
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        assert features[col].map(type).eq(str).all()


def test_delay_features_match_python(features):
    pay = features[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
    np.testing.assert_array_equal(features["DELAY_MAX"], pay.max(axis=1))
    np.testing.assert_array_equal(features["DELAY_MONTHS"], (pay >= 1).sum(axis=1))


def test_utilization_matches_python(features):
    expected = features["BILL_AMT1"] / features["LIMIT_BAL"]
    np.testing.assert_allclose(features["UTIL_LATEST"], expected)


def test_normalize_renames_and_rejects_missing_column():
    source = make_source(50)
    out = normalize(source)
    assert {"CLIENT_ID", "TARGET"} <= set(out.columns)
    with pytest.raises(SchemaError, match="PAY_AMT3"):
        normalize(source.drop(columns="PAY_AMT3"))
    with pytest.raises(SchemaError, match="binary"):
        normalize(source.assign(**{SOURCE_TARGET: 2}))


def test_header_row_is_found_below_code_row(tmp_path, monkeypatch):
    source = make_source(20)
    codes = pd.DataFrame([["", *[f"X{i}" for i in range(1, 24)], "Y"]], columns=source.columns)
    sheet = pd.concat([pd.DataFrame([source.columns], columns=source.columns), source.astype(object)])
    sheet = pd.concat([codes, sheet], ignore_index=True)

    def fake_read_excel(path, header=None, nrows=None, engine=None):
        raw = sheet.head(nrows) if nrows else sheet
        if header is None:
            return raw.reset_index(drop=True)
        body = raw.iloc[header + 1 :].reset_index(drop=True)
        body.columns = raw.iloc[header].tolist()
        return body.apply(pd.to_numeric)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    out = read_spreadsheet(tmp_path / "sheet.xls")
    assert len(out) == 20
    assert out["CLIENT_ID"].tolist() == list(range(1, 21))
