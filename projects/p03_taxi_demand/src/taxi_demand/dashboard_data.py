"""Filters and measures behind the dashboard, kept free of Streamlit so that tests can call them."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MART_NAMES = ["fact_hourly", "zone_monthly", "daily", "quality_audit", "zones"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Filters:
    start: date
    end: date
    boroughs: tuple[str, ...]
    payments: tuple[str, ...]


def load_marts(marts_dir: Path) -> dict[str, pd.DataFrame]:
    marts = {name: pd.read_parquet(marts_dir / f"{name}.parquet") for name in MART_NAMES}
    for name in ["fact_hourly", "daily"]:
        marts[name]["pickup_date"] = pd.to_datetime(marts[name]["pickup_date"]).dt.date
    return marts


def apply_filters(fact: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    mask = (
        (fact["pickup_date"] >= filters.start)
        & (fact["pickup_date"] <= filters.end)
        & fact["pickup_borough"].isin(filters.boroughs)
        & fact["payment"].isin(filters.payments)
    )
    return fact.loc[mask]


def previous_period(filters: Filters) -> Filters:
    """The window of the same length that ends the day before ``filters.start``."""
    length = (filters.end - filters.start).days + 1
    end = filters.start - timedelta(days=1)
    return Filters(end - timedelta(days=length - 1), end, filters.boroughs, filters.payments)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def kpis(fact: pd.DataFrame) -> dict[str, float]:
    """Headline measures. Every ratio is a ratio of sums, so it stays exact under any filter."""
    card = fact[fact["payment"] == "Credit card"]
    days = fact["pickup_date"].nunique()
    return {
        "trips": float(fact["trips"].sum()),
        "trips_per_day": _ratio(fact["trips"].sum(), days),
        "revenue_usd": float(fact["total_usd"].sum()),
        "mean_fare_usd": _ratio(fact["fare_usd"].sum(), fact["trips"].sum()),
        "card_tip_rate": _ratio(card["tip_usd"].sum(), card["fare_usd"].sum()),
        "mean_duration_min": _ratio(fact["duration_min"].sum(), fact["trips"].sum()),
        "mean_speed_mph": _ratio(fact["distance_mi"].sum(), fact["duration_min"].sum() / 60),
        "airport_share": _ratio(fact["airport_trips"].sum(), fact["trips"].sum()),
    }


def relative_change(current: float, previous: float) -> float:
    if not previous or np.isnan(previous) or np.isnan(current):
        return float("nan")
    return current / previous - 1


def daily_series(fact: pd.DataFrame) -> pd.DataFrame:
    out = fact.groupby("pickup_date", as_index=False)[["trips", "total_usd"]].sum().sort_values("pickup_date")
    out["trips_7d"] = out["trips"].rolling(7, min_periods=7, center=True).mean()
    return out


def monthly_by_payment(fact: pd.DataFrame) -> pd.DataFrame:
    return fact.groupby(["pickup_month", "payment"], as_index=False)[["trips", "total_usd"]].sum()


def hour_weekday(fact: pd.DataFrame) -> pd.DataFrame:
    """Mean trips per hour, for each weekday and hour, over the days in the filtered window."""
    days = fact.groupby("pickup_dow")["pickup_date"].nunique()
    grid = fact.groupby(["pickup_dow", "pickup_hour"])["trips"].sum().unstack(fill_value=0)
    grid = grid.div(days, axis=0).reindex(index=range(1, 8), columns=range(24), fill_value=0)
    grid.index = WEEKDAYS
    return grid


def speed_by_hour(fact: pd.DataFrame) -> pd.DataFrame:
    frame = fact.assign(day_type=np.where(fact["pickup_dow"] <= 5, "Weekday", "Weekend"))
    out = frame.groupby(["day_type", "pickup_hour"], as_index=False)[["distance_mi", "duration_min"]].sum()
    out["mean_speed_mph"] = out["distance_mi"] / (out["duration_min"] / 60)
    return out


def borough_summary(fact: pd.DataFrame) -> pd.DataFrame:
    out = fact.groupby("pickup_borough", as_index=False)[["trips", "total_usd"]].sum()
    out["share"] = out["trips"] / out["trips"].sum()
    return out.sort_values("trips", ascending=False)


def zone_ranking(zone_monthly: pd.DataFrame, filters: Filters, top: int = 15) -> pd.DataFrame:
    """Top pickup zones in the filtered months and boroughs. The zone table has no payment split."""
    months = range(filters.start.month, filters.end.month + 1)
    part = zone_monthly[
        zone_monthly["pickup_month"].isin(months) & zone_monthly["borough"].isin(filters.boroughs)
    ]
    out = part.groupby(["zone", "borough"], as_index=False)[
        ["trips", "total_usd", "card_tip_usd", "card_fare_usd"]
    ].sum()
    out["share"] = out["trips"] / out["trips"].sum()
    out["mean_total_usd"] = out["total_usd"] / out["trips"]
    out["card_tip_rate"] = out["card_tip_usd"] / out["card_fare_usd"]
    return out.sort_values("trips", ascending=False).head(top)


def weather_view(daily: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Non holiday days in the window. The demand index is computed over all boroughs and payments."""
    mask = (
        (daily["pickup_date"] >= filters.start) & (daily["pickup_date"] <= filters.end) & ~daily["is_holiday"]
    )
    return daily.loc[mask]
