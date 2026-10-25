import json
import zipfile

import pytest

from inventory_rl import data
from inventory_rl.config import FIT_START, PROFILES, TEST_DAYS, TEST_START


def test_three_products_with_their_profiles(prepared):
    products = data.load_products(prepared)
    assert [p["product"] for p in products] == list(PROFILES)
    assert len({p["series_id"] for p in products}) == 3
    for product in products:
        low, high = PROFILES[product["product"]]["zero_share"]
        assert low <= product["selection"]["zero_share"] <= high
        assert product["unit_cost"] < product["mean_price"]
        assert product["capacity"] >= 20
        assert len(product["demand"]["weekday_mean"]) == 7
        assert {"fit", "test"} <= set(product["fit_check"])


def test_demand_table_covers_the_fit_window_and_test_year(prepared):
    demand = data.load_demand(prepared)
    for _, part in demand.groupby("product"):
        assert part["d"].min() == FIT_START
        assert (part["d"] >= TEST_START).sum() == TEST_DAYS
        assert part["sell_price"].notna().all()
        assert part.loc[part["closed"], "units"].eq(0).all()


def test_sample_reads_only_the_sample_store(archive, tmp_path):
    products = data.write(archive, tmp_path, sample=0.1)
    assert {p["store_id"] for p in products} == {"CA_1"}
    assert json.loads((tmp_path / "products.json").read_text())["stores"] == ["CA_1"]


def test_missing_file_is_reported(archive, tmp_path):
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(broken, "w") as target:
        for name in source.namelist():
            if name != "sales_test_evaluation.csv":
                target.writestr(name, source.read(name))
    with pytest.raises(data.SchemaError, match="sales_test_evaluation.csv"):
        data.write(broken, tmp_path / "out")


def test_wrong_checksum_is_refused(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "m5.zip").write_bytes(b"not the archive")
    with pytest.raises(data.SchemaError, match="SHA256"):
        data.download(raw)
