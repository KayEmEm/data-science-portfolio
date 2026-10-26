"""Run the analysis query library against the warehouse and write report tables, key numbers and figures."""

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from taxi_demand import config
from taxi_demand.warehouse import analysis_queries, run_query

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


def key_numbers(results: dict[str, pd.DataFrame], daily: pd.DataFrame) -> dict[str, float]:
    """The figures the findings report quotes, computed in one place."""
    audit = results["a10_quality_audit"].set_index("check_result")
    zones = results["a05_zone_concentration"]
    airport = results["a07_airport_trips"].set_index("trip_type")
    speed = results["a03_speed_by_hour"]
    weekday_speed = speed[speed["day_type"] == "Weekday"].set_index("hour")["mean_speed_mph"]
    weather = results["a04_weather_effect"].set_index("weather_class")["mean_demand_index"]
    hourly = results["a02_hour_weekday"]
    peak = hourly.loc[hourly["mean_trips"].idxmax()]
    trough = hourly.loc[hourly["mean_trips"].idxmin()]
    monthly = results["a01_monthly_trend"]
    return {
        "source_rows": int(audit["trips"].sum()),
        "usable_trips": int(audit.loc["kept", "trips"]),
        "usable_share": float(audit.loc["kept", "share"]),
        "days": int(len(daily)),
        "mean_trips_per_day": float(daily["trips"].mean()),
        "total_revenue_usd": float(daily["total_usd"].sum()),
        "busiest_month": int(monthly.loc[monthly["trips_per_day"].idxmax(), "month"]),
        "quietest_month": int(monthly.loc[monthly["trips_per_day"].idxmin(), "month"]),
        "peak_weekday": WEEKDAYS[int(peak["weekday"]) - 1],
        "peak_hour": int(peak["hour"]),
        "peak_mean_trips": float(peak["mean_trips"]),
        "trough_weekday": WEEKDAYS[int(trough["weekday"]) - 1],
        "trough_hour": int(trough["hour"]),
        "trough_mean_trips": float(trough["mean_trips"]),
        "zones_with_trips": int(len(zones)),
        "top10_zone_share": float(
            zones.loc[zones["zone_rank"] == min(10, len(zones)), "cumulative_share"].iloc[0]
        ),
        "airport_trip_share": float(airport.loc["Airport", "share_of_trips"]),
        "airport_revenue_share": float(airport.loc["Airport", "share_of_revenue"]),
        "weekday_speed_slowest_mph": float(weekday_speed.min()),
        "weekday_speed_slowest_hour": int(weekday_speed.idxmin()),
        "weekday_speed_fastest_mph": float(weekday_speed.max()),
        "weekday_speed_fastest_hour": int(weekday_speed.idxmax()),
        "card_tip_rate": float(
            (daily["card_tip_usd"].sum() / daily["card_fare_usd"].sum())
            if daily["card_fare_usd"].sum()
            else np.nan
        ),
        "demand_index_by_weather": {k: float(v) for k, v in weather.items()},
    }


def plot_daily(daily: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    dates = pd.to_datetime(daily["pickup_date"])
    ax.plot(dates, daily["trips"], color="#b7d3f6", linewidth=1, label="Daily trips")
    ax.plot(dates, daily["trips"].rolling(7, center=True).mean(), color=BLUE, linewidth=2, label="7 day mean")
    ax.set_ylabel("Trips per day")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Daily yellow taxi trips")
    fig.savefig(out)
    plt.close(fig)


def plot_heatmap(hourly: pd.DataFrame, out: Path) -> None:
    grid = hourly.pivot(index="weekday", columns="hour", values="mean_trips").reindex(range(1, 8))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), WEEKDAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of pickup")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Mean trips per hour")
    ax.set_title("Demand by weekday and hour")
    fig.savefig(out)
    plt.close(fig)


def plot_weather(daily: pd.DataFrame, out: Path) -> None:
    order = ["dry, under 1 mm", "rain 1 to 10 mm", "rain 10 mm or more", "snow 10 mm or more"]
    usable = daily[~daily["is_holiday"]]
    groups = [(c, usable.loc[usable["weather_class"] == c, "demand_index"].dropna()) for c in order]
    groups = [(c, g) for c, g in groups if len(g)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (_, values) in enumerate(groups):
        jitter = np.random.default_rng(config.SEED).uniform(-0.15, 0.15, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=10, color="#9ec5f4", linewidths=0)
        ax.plot([i - 0.25, i + 0.25], [values.mean()] * 2, color=BLUE, linewidth=2.5)
    ax.axhline(1.0, color="#8a8984", linewidth=0.8)
    ax.set_xticks(range(len(groups)), [f"{c}\n({len(g)} days)" for c, g in groups])
    ax.set_ylabel("Demand index (1.0 = same weekday mean)")
    ax.set_title("Daily demand index by Central Park weather (holidays left out)")
    fig.savefig(out)
    plt.close(fig)


def plot_zone_concentration(zones: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(zones["zone_rank"], zones["cumulative_share"] * 100, color=BLUE, linewidth=2)
    ax.set_xlabel("Pickup zones, ranked by trips")
    ax.set_ylabel("Cumulative share of trips (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Concentration of pickups across zones")
    fig.savefig(out)
    plt.close(fig)


def plot_speed(speed: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for day_type, color in [("Weekday", BLUE), ("Weekend", ORANGE)]:
        part = speed[speed["day_type"] == day_type]
        ax.plot(part["hour"], part["mean_speed_mph"], color=color, linewidth=2, label=day_type)
    ax.set_xlabel("Hour of pickup")
    ax.set_ylabel("Mean speed (mph)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False)
    ax.set_title("Trip speed by hour")
    fig.savefig(out)
    plt.close(fig)


def write_tables(results: dict[str, pd.DataFrame], out: Path) -> None:
    sections = [
        "# Query results",
        "",
        "Each table is the output of one query in `src/taxi_demand/sql/analysis/`.",
        "",
    ]
    for name, frame in results.items():
        shown = frame if len(frame) <= 40 else frame.head(25)
        sections += [f"## {name}", "", shown.to_markdown(index=False, floatfmt=",.4g"), ""]
        if len(frame) > len(shown):
            sections += [f"First {len(shown)} of {len(frame)} rows.", ""]
    out.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=config.WAREHOUSE_PATH)
    parser.add_argument("--reports", type=Path, default=config.REPORTS_DIR)
    args = parse_args(parser, __file__)

    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.warehouse), read_only=True) as con:
        results = {name: run_query(con, name) for name in analysis_queries()}
        daily = con.table("daily").df()

    apply_style()
    plot_daily(daily, figures / "daily_demand.png")
    plot_heatmap(results["a02_hour_weekday"], figures / "hour_weekday.png")
    plot_weather(daily, figures / "weather_effect.png")
    plot_zone_concentration(results["a05_zone_concentration"], figures / "zone_concentration.png")
    plot_speed(results["a03_speed_by_hour"], figures / "speed_by_hour.png")
    write_tables(results, args.reports / "query_results.md")
    numbers = key_numbers(results, daily)
    (args.reports / "key_numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
