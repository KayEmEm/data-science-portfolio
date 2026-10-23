"""Paths, sources and code tables for project 3."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRIPS_DIR = RAW_DIR / "trips"
ZONES_PATH = RAW_DIR / "taxi_zone_lookup.csv"
WEATHER_PATH = RAW_DIR / "weather_central_park.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
WAREHOUSE_PATH = ARTIFACTS_DIR / "taxi.duckdb"
MARTS_DIR = ARTIFACTS_DIR / "marts"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

YEAR = 2024
SEED = 42

# NYC Taxi and Limousine Commission trip record files, one Parquet file per month.
TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# NOAA NCEI daily summaries (GHCN Daily) for Central Park, metric units.
WEATHER_STATION = "USW00094728"
WEATHER_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
    "&stations={station}&startDate={year}-01-01&endDate={year}-12-31"
    "&dataTypes=PRCP,SNOW,TMAX,TMIN&units=metric&format=csv"
)

# Source columns the pipeline reads. The loader checks them before any query runs.
TRIP_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount",
]
ZONE_COLUMNS = ["LocationID", "Borough", "Zone", "service_zone"]
WEATHER_COLUMNS = ["DATE", "PRCP", "SNOW", "TMAX", "TMIN"]

# Days left out of the weather comparison: United States federal holidays plus these
# two days, on which demand is atypical for reasons unrelated to weather. An analysis choice.
EXTRA_HOLIDAYS = ["12-24", "12-31"]

# Limits for a plausible trip. Trips outside them are counted by the audit and
# excluded from the analysis tables. They are analysis choices, not TLC rules.
MAX_DURATION_MIN = 180
MAX_DISTANCE_MI = 100
MAX_FARE_USD = 500
MIN_SPEED_MPH = 0.5
MAX_SPEED_MPH = 80
