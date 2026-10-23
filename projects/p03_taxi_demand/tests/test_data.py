import pandas as pd
import pytest

from taxi_demand.config import TRIP_COLUMNS, WEATHER_COLUMNS
from taxi_demand.data import SchemaError, check_csv, check_trip_file, download_all, missing_columns


def test_missing_columns_ignores_case():
    assert missing_columns(["vendorid", "Airport_fee"], ["VendorID", "airport_fee", "fare_amount"]) == [
        "fare_amount"
    ]


def test_synthetic_files_pass_schema_checks(raw_dir):
    download_all(raw_dir, 2024, [1, 2, 3], skip_download=True)


def test_trip_file_without_a_required_column_is_rejected(raw_dir, tmp_path):
    source = next((raw_dir / "trips").glob("*.parquet"))
    broken = tmp_path / "broken.parquet"
    pd.read_parquet(source).drop(columns=[TRIP_COLUMNS[-1]]).to_parquet(broken)
    with pytest.raises(SchemaError, match=TRIP_COLUMNS[-1]):
        check_trip_file(broken)


def test_weather_csv_without_precipitation_is_rejected(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({c: [1] for c in WEATHER_COLUMNS if c != "PRCP"}).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="PRCP"):
        check_csv(path, WEATHER_COLUMNS)
