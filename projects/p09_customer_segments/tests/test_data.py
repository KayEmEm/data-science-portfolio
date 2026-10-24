import zipfile

import pandas as pd
import pytest

from customer_segments.categories import CATEGORIES, OTHER, categorize, categorize_series
from customer_segments.data import SchemaError, extract, normalize, prepare, read_raw
from customer_segments.synthetic import write_xlsx


@pytest.mark.parametrize(
    ("description", "category"),
    [
        ("WHITE HANGING HEART T-LIGHT HOLDER", "lights_candles"),
        ("CHRISTMAS T-LIGHT HOLDER", "christmas"),
        ("JUMBO BAG RED RETROSPOT", "bags"),
        ("REGENCY CAKESTAND 3 TIER", "kitchen_dining"),
        ("SPRING FLOWER CHAIN", "garden_outdoor"),
        ("WIDGET", OTHER),
    ],
)
def test_category_rules(description, category):
    assert categorize(description) == category


def test_category_series_matches_single_calls():
    values = pd.Series(["ENAMEL MUG", None, "ENAMEL MUG", "DOOR MAT RED SPOT"])
    assert categorize_series(values).tolist() == ["kitchen_dining", OTHER, "kitchen_dining", "home_decor"]
    assert set(categorize_series(values)) <= set(CATEGORIES)


def test_cleaning_counts(csv_folder, raw_lines):
    lines, audit = prepare(csv_folder)
    assert audit["rows_read"] == len(raw_lines)
    assert audit["duplicate_rows"] >= int(0.01 * len(raw_lines) * 0.9)
    assert audit["rows_without_customer"] > 0
    assert lines["customer_id"].notna().all()
    returns = lines.loc[lines["is_return"]]
    assert len(returns) > 0 and (returns["quantity"] < 0).all()
    assert (lines.loc[~lines["is_return"], ["quantity", "price"]] > 0).all().all()
    assert set(lines.loc[~lines["is_product"], "stock_code"]) == {"POST"}
    assert set(lines.loc[lines["is_product"], "category"]) <= set(CATEGORIES)


def test_workbook_with_overlapping_sheets_and_zip(raw_lines, tmp_path):
    path = write_xlsx(tmp_path / "xlsx", raw_lines.iloc[::20])
    raw = read_raw(path)
    assert raw["sheet"].nunique() == 2
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(path, arcname="sub/online_retail_II.xlsx")
    assert extract(archive, tmp_path / "out").name == "online_retail_II.xlsx"
    lines, audit = prepare(path)
    assert audit["rows_in_two_sheets"] > 0
    assert audit["duplicate_rows"] >= audit["rows_in_two_sheets"]


def test_missing_column_is_reported(raw_lines):
    with pytest.raises(SchemaError):
        normalize(raw_lines.drop(columns=["Price"]).assign(sheet="x"))
