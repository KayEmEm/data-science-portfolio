import zipfile

import numpy as np
import pytest

from retail_forecast import data
from retail_forecast.data import SchemaError, normalize


def copy(tables):
    return {k: v.copy() for k, v in tables.items()}


def test_normalize_joins_train_and_test_days(raw_tables):
    out = normalize(copy(raw_tables), expected_series=None)
    assert out["sales"].shape == (len(raw_tables["train"]), 400)
    assert list(out["sales"].columns[:2]) == ["d_1", "d_2"]
    assert out["prices"].shape == out["sales"].shape
    assert out["calendar"]["d"].tolist() == list(range(1, 401))


def test_daily_prices_follow_the_week_and_start_at_release(raw_tables):
    out = normalize(copy(raw_tables), expected_series=None)
    prices = out["prices"].to_numpy()
    sales = out["sales"].to_numpy()
    assert not (np.isnan(prices) & (sales > 0)).any()
    weeks = out["calendar"]["wm_yr_wk"].to_numpy()
    same_week = weeks[1:] == weeks[:-1]
    both = ~np.isnan(prices[:, 1:]) & ~np.isnan(prices[:, :-1])
    assert np.allclose(prices[:, 1:][both & same_week], prices[:, :-1][both & same_week])


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda t: t["train"].drop(columns=["store_id"], inplace=True), "lack"),
        (lambda t: t["train"].__setitem__("d_5", -1), "negative"),
        (
            lambda t: t["test"].__setitem__("item_id", t["test"]["item_id"].iloc[::-1].to_numpy()),
            "different order",
        ),
        (lambda t: t["weights"].__setitem__("weight", t["weights"]["weight"] * 2), "sum to one"),
        (lambda t: t["prices"].drop(index=t["prices"].index[:40], inplace=True), "without a price"),
    ],
)
def test_normalize_rejects_bad_tables(raw_tables, corrupt, message):
    tables = copy(raw_tables)
    corrupt(tables)
    with pytest.raises(SchemaError, match=message):
        normalize(tables, expected_series=None)


def test_normalize_checks_the_m5_size(raw_tables):
    with pytest.raises(SchemaError, match="expected 30490 series"):
        normalize(copy(raw_tables))


def test_download_rejects_a_file_with_the_wrong_checksum(tmp_path):
    with zipfile.ZipFile(tmp_path / data.RAW_NAME, "w") as archive:
        archive.writestr("calendar.csv", "not the file")
    with pytest.raises(SchemaError, match="SHA256"):
        data.download(tmp_path)


def test_load_returns_matrices(tables):
    assert tables["sales"].dtype == np.int32
    assert tables["prices"].dtype == np.float32
    assert tables["sales"].shape == (54, 500)
