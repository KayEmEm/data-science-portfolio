"""Synthetic trip, zone and weather files with the TLC and NOAA source schemas, for tests and smoke runs only.

The values are random. Results computed on them say nothing about New York taxi demand
and must never be reported as project results.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from taxi_demand.config import SEED, WEATHER_STATION

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
HOUR_WEIGHTS = np.array(
    [3, 2, 1.5, 1, 1, 1.5, 3, 5, 6, 6, 6, 6, 6.5, 6.5, 7, 7.5, 8, 9, 9, 8, 7, 6.5, 5.5, 4.5]
)


def zone_table() -> pd.DataFrame:
    """265 zones as in the TLC lookup: 1 to 263 real, 264 unknown, 265 outside the city."""
    ids = np.arange(1, 266)
    borough = [BOROUGHS[(i * 7) % 5] for i in ids]
    borough[0] = "EWR"
    borough[263], borough[264] = "Unknown", "N/A"
    return pd.DataFrame(
        {
            "LocationID": ids,
            "Borough": borough,
            "Zone": [f"Synthetic zone {i}" for i in ids],
            "service_zone": ["Yellow Zone" if b == "Manhattan" else "Boro Zone" for b in borough],
        }
    )


def weather_table(year: int, rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    season = np.cos(2 * np.pi * (days.day_of_year - 200) / 365)
    tmax = 18 + 12 * season + rng.normal(0, 3, len(days))
    wet = rng.uniform(size=len(days)) < 0.3
    precip = np.where(wet, rng.gamma(1.2, 8, len(days)), 0).round(1)
    snow = np.where(wet & (tmax < 3), precip * 10, 0).round(0)
    return pd.DataFrame(
        {
            "STATION": WEATHER_STATION,
            "DATE": days.strftime("%Y-%m-%d"),
            "PRCP": precip,
            "SNOW": snow,
            "TMAX": tmax.round(1),
            "TMIN": (tmax - 8).round(1),
        }
    )


def trips_for_month(
    year: int, month: int, n: int, weather: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    precip = weather.set_index("DATE").loc[days.strftime("%Y-%m-%d"), "PRCP"].to_numpy()
    day_weight = np.where(days.day_of_week >= 5, 0.85, 1.0) * np.where(precip >= 10, 0.9, 1.0)
    day = rng.choice(len(days), n, p=day_weight / day_weight.sum())
    hour = rng.choice(24, n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    pickup = (
        pd.to_datetime(days[day])
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    )
    distance = rng.gamma(1.6, 1.9, n).round(2)
    speed = np.clip(rng.normal(12, 3, n) - 3 * ((hour >= 8) & (hour <= 18)), 3, 40)
    duration_s = (distance / speed * 3600).astype(int) + 60
    zone_weights = 1 / np.arange(1, 264) ** 1.1
    pu = rng.choice(np.arange(1, 264), n, p=zone_weights / zone_weights.sum())
    do = rng.choice(np.arange(1, 264), n, p=zone_weights / zone_weights.sum())
    payment = rng.choice([1, 2, 0, 3, 4], n, p=[0.74, 0.18, 0.04, 0.02, 0.02])
    fare = (3 + 2.8 * distance + 0.6 * duration_s / 60).round(2)
    tip = np.where(payment == 1, (fare * rng.uniform(0.1, 0.3, n)).round(2), 0.0)
    tolls = np.where(rng.uniform(size=n) < 0.05, 6.94, 0.0)
    frame = pd.DataFrame(
        {
            "VendorID": rng.choice([1, 2], n).astype("int32"),
            "tpep_pickup_datetime": pickup.astype("datetime64[us]"),
            "tpep_dropoff_datetime": (pickup + pd.to_timedelta(duration_s, unit="s")).astype(
                "datetime64[us]"
            ),
            "passenger_count": rng.choice([1, 1, 1, 2, 3], n).astype("float64"),
            "trip_distance": distance,
            "RatecodeID": np.ones(n),
            "store_and_fwd_flag": "N",
            "PULocationID": pu.astype("int32"),
            "DOLocationID": do.astype("int32"),
            "payment_type": payment.astype("int64"),
            "fare_amount": fare,
            "extra": 1.0,
            "mta_tax": 0.5,
            "tip_amount": tip,
            "tolls_amount": tolls,
            "improvement_surcharge": 1.0,
            "total_amount": (fare + tip + tolls + 2.5).round(2),
            "congestion_surcharge": 2.5,
            "Airport_fee": 0.0,
        }
    )
    # A few rows of each defect the quality audit looks for.
    bad = rng.choice(n, 6 * max(1, n // 500), replace=False).reshape(6, -1)
    frame.loc[bad[0], "trip_distance"] = 0.0
    frame.loc[bad[1], "tpep_dropoff_datetime"] = frame.loc[bad[1], "tpep_pickup_datetime"]
    frame.loc[bad[2], "fare_amount"] = -frame.loc[bad[2], "fare_amount"]
    frame.loc[bad[3], "PULocationID"] = 264
    frame.loc[bad[4], "tpep_pickup_datetime"] = pd.Timestamp(f"{year - 1}-12-31 23:00")
    frame.loc[bad[5], "tpep_dropoff_datetime"] += pd.Timedelta(hours=5)
    return frame


def write_raw(out_dir: Path, year: int = 2024, months: int = 12, trips_per_month: int = 4000) -> None:
    """Write files laid out as ``make data`` leaves them under data/raw."""
    rng = np.random.default_rng(SEED)
    (out_dir / "trips").mkdir(parents=True, exist_ok=True)
    zone_table().to_csv(out_dir / "taxi_zone_lookup.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    weather = weather_table(year, rng)
    weather.to_csv(out_dir / "weather_central_park.csv", index=False, quoting=csv.QUOTE_ALL)
    for month in range(1, months + 1):
        trips = trips_for_month(year, month, trips_per_month, weather, rng)
        trips.to_parquet(out_dir / "trips" / f"yellow_tripdata_{year}-{month:02d}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output folder, laid out like data/raw")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--trips-per-month", type=int, default=4000)
    args = parser.parse_args()
    write_raw(args.out, args.year, args.months, args.trips_per_month)


if __name__ == "__main__":
    main()
