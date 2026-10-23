"""Taxi operations dashboard: a mock up in the layout of a BI report, built with Streamlit and Plotly.

Run from the project folder: ``make dashboard``, or
``streamlit run dashboard/app.py -- --marts artifacts/marts``.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from taxi_demand import config
from taxi_demand.dashboard_data import (
    Filters,
    apply_filters,
    borough_summary,
    daily_series,
    hour_weekday,
    kpis,
    load_marts,
    monthly_by_payment,
    previous_period,
    relative_change,
    speed_by_hour,
    weather_view,
    zone_ranking,
)

# Categorical slots in fixed order; each payment group keeps its color under any filter.
PAYMENT_COLORS = {"Credit card": "#2a78d6", "Cash": "#eb6834", "Other": "#1baf7a"}
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#b7d3f6", "#eb6834"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 1rem; max-width: 1500px;}
.report-header {display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px solid #e6e5e0; padding-bottom: 0.4rem; margin-bottom: 0.8rem;}
.report-header .title {font-size: 1.45rem; font-weight: 650; color: #0b0b0b;}
.report-header span {color: #52514e; font-size: 0.85rem;}
.kpi {background: #fcfcfb; border: 1px solid #e6e5e0; border-radius: 8px; padding: 0.7rem 0.9rem;}
.kpi .label {color: #52514e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;}
.kpi .value {color: #0b0b0b; font-size: 1.55rem; font-weight: 600; margin-top: 0.15rem;}
.kpi .delta {color: #52514e; font-size: 0.78rem; margin-top: 0.1rem;}
.panel-title {font-weight: 600; font-size: 0.95rem; margin: 0.2rem 0 0 0;}
.panel-note {color: #52514e; font-size: 0.78rem; margin: 0 0 0.2rem 0;}
</style>
"""


def marts_dir() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marts", type=Path, default=Path(os.environ.get("TAXI_MARTS", config.MARTS_DIR)))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.marts


@st.cache_data
def cached_marts(path: str):
    return load_marts(Path(path))


def layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", color=INK_MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=INK_MUTED)
    return fig


def panel(title: str, note: str = "") -> None:
    st.markdown(f'<p class="panel-title">{title}</p>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="panel-note">{note}</p>', unsafe_allow_html=True)


def fmt_number(value: float, kind: str) -> str:
    if math.isnan(value):
        return "n/a"
    if kind == "count":
        return f"{value / 1e6:.2f} M" if value >= 1e6 else f"{value:,.0f}"
    if kind == "usd_large":
        return f"${value / 1e6:,.1f} M" if value >= 1e6 else f"${value:,.0f}"
    if kind == "usd":
        return f"${value:,.2f}"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "min":
        return f"{value:.1f} min"
    if kind == "mph":
        return f"{value:.1f} mph"
    return f"{value:,.0f}"


def kpi_tile(column, label: str, value: float, previous: float, kind: str) -> None:
    change = relative_change(value, previous)
    delta = "no prior period in data" if math.isnan(change) else f"{change:+.1%} vs prior period"
    column.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{fmt_number(value, kind)}</div><div class="delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_filters(fact) -> Filters:
    first, last = min(fact["pickup_date"]), max(fact["pickup_date"])
    st.sidebar.header("Filters")
    picked = st.sidebar.date_input("Pickup dates", (first, last), min_value=first, max_value=last)
    start, end = (picked[0], picked[1]) if len(picked) == 2 else (picked[0], picked[0])
    boroughs_all = sorted(fact["pickup_borough"].unique())
    boroughs = st.sidebar.multiselect("Pickup borough", boroughs_all, default=boroughs_all)
    payments_all = [p for p in PAYMENT_COLORS if p in set(fact["payment"])]
    payments = st.sidebar.multiselect("Payment", payments_all, default=payments_all)
    st.sidebar.caption(
        "Source: NYC TLC yellow taxi trip records and NOAA daily weather for Central Park. "
        "Trips that fail a quality check are excluded; see the Data quality tab."
    )
    return Filters(start, end, tuple(boroughs), tuple(payments))


def overview_tab(fact) -> None:
    left, right = st.columns([2, 1])
    with left:
        panel("Daily trips", "Light line: each day. Dark line: centered 7 day mean.")
        series = daily_series(fact)
        fig = go.Figure()
        fig.add_scatter(
            x=series["pickup_date"], y=series["trips"], name="Daily", line={"color": BLUE_LIGHT, "width": 1}
        )
        fig.add_scatter(
            x=series["pickup_date"], y=series["trips_7d"], name="7 day mean", line={"color": BLUE, "width": 2}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(layout(fig), use_container_width=True)
    with right:
        panel("Trips by pickup borough", "Share of filtered trips.")
        boroughs = borough_summary(fact)
        fig = go.Figure(
            go.Bar(
                x=boroughs["trips"],
                y=boroughs["pickup_borough"],
                orientation="h",
                marker={"color": BLUE, "cornerradius": 4},
                text=[f"{s:.1%}" for s in boroughs["share"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,.0f} trips<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig.update_xaxes(range=[0, boroughs["trips"].max() * 1.2])
        st.plotly_chart(layout(fig), use_container_width=True)

    panel(
        "Monthly trips by payment",
        "Cash tips are not recorded in the source, so tip measures use card trips only.",
    )
    monthly = monthly_by_payment(fact)
    fig = go.Figure()
    for payment, color in PAYMENT_COLORS.items():
        part = monthly[monthly["payment"] == payment]
        if len(part):
            fig.add_bar(x=part["pickup_month"], y=part["trips"], name=payment, marker={"color": color})
    fig.update_layout(barmode="stack", bargap=0.3, legend_traceorder="normal")
    fig.update_traces(marker_line={"color": "white", "width": 2})
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    st.plotly_chart(layout(fig, 280), use_container_width=True)


def time_tab(fact) -> None:
    panel(
        "Mean trips per hour", "Average over the days in the filtered window, by weekday and hour of pickup."
    )
    grid = hour_weekday(fact)
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[f"{h:02d}" for h in grid.columns],
            y=list(grid.index),
            colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            xgap=2,
            ygap=2,
            colorbar={"title": "Trips", "thickness": 10},
            hovertemplate="%{y} %{x}:00: %{z:,.0f} trips per hour<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(layout(fig, 300), use_container_width=True)

    panel("Mean trip speed by hour", "Total distance over total time; a measure of street congestion.")
    speed = speed_by_hour(fact)
    fig = go.Figure()
    for day_type, color in [("Weekday", BLUE), ("Weekend", ORANGE)]:
        part = speed[speed["day_type"] == day_type]
        fig.add_scatter(
            x=part["pickup_hour"], y=part["mean_speed_mph"], name=day_type, line={"color": color, "width": 2}
        )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero", title="mph")
    fig.update_xaxes(title="Hour of pickup", dtick=3)
    st.plotly_chart(layout(fig, 280), use_container_width=True)


def zones_tab(marts, filters: Filters) -> None:
    ranking = zone_ranking(marts["zone_monthly"], filters)
    left, right = st.columns([1, 1])
    with left:
        panel(
            "Top 15 pickup zones",
            "Months and boroughs follow the filters; the zone table has no payment split.",
        )
        fig = go.Figure(
            go.Bar(
                x=ranking["trips"],
                y=ranking["zone"],
                orientation="h",
                marker={"color": BLUE, "cornerradius": 4},
                customdata=ranking[["borough", "share"]],
                hovertemplate="%{y} (%{customdata[0]}): %{x:,.0f} trips, %{customdata[1]:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(layout(fig, 460), use_container_width=True)
    with right:
        panel("Zone table")
        table = ranking[["zone", "borough", "trips", "share", "card_tip_rate"]].rename(
            columns={
                "zone": "Zone",
                "borough": "Borough",
                "trips": "Trips",
                "share": "Share",
                "card_tip_rate": "Card tip rate",
            }
        )
        st.dataframe(
            table.style.format({"Trips": "{:,.0f}", "Share": "{:.1%}", "Card tip rate": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
            height=460,
        )


def weather_tab(marts, filters: Filters) -> None:
    days = weather_view(marts["daily"], filters)
    panel(
        "Demand index by weather",
        "Each dot is a day. 1.0 equals the mean of the same weekday in the same month; "
        "all boroughs and payments.",
    )
    order = ["dry, under 1 mm", "rain 1 to 10 mm", "rain 10 mm or more", "snow 10 mm or more", "no record"]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        present = [c for c in order if (days["weather_class"] == c).any()]
        for weather_class in present:
            part = days[days["weather_class"] == weather_class]
            fig.add_box(
                y=part["demand_index"],
                name=f"{weather_class}<br>{len(part)} days",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"color": BLUE, "size": 5, "opacity": 0.55},
                line={"color": BLUE, "width": 1.5},
                fillcolor="rgba(0,0,0,0)",
                customdata=part["pickup_date"].astype(str),
                hovertemplate="%{customdata}: index %{y:.2f}<extra></extra>",
            )
        fig.add_hline(y=1.0, line={"color": INK_MUTED, "width": 1})
        fig.update_layout(showlegend=False)
        st.plotly_chart(layout(fig, 340), use_container_width=True)
    with right:
        fig = go.Figure(
            go.Scatter(
                x=days["tmax_c"],
                y=days["trips"],
                mode="markers",
                marker={"color": BLUE, "size": 7, "opacity": 0.6, "line": {"color": "white", "width": 1}},
                customdata=days["pickup_date"].astype(str),
                hovertemplate="%{customdata}: %{x:.1f} C, %{y:,.0f} trips<extra></extra>",
            )
        )
        fig.update_xaxes(title="Maximum temperature (C)")
        fig.update_yaxes(title="Trips per day", rangemode="tozero")
        st.plotly_chart(layout(fig, 340), use_container_width=True)


def quality_tab(marts) -> None:
    audit = marts["quality_audit"]
    kept = audit.loc[audit["check_result"] == "kept"]
    kept_share = float(kept["share"].iloc[0]) if len(kept) else float("nan")
    a, b = st.columns(2)
    a.metric("Source rows", f"{audit['trips'].sum():,.0f}")
    b.metric("Kept for analysis", f"{kept_share:.1%}")
    panel(
        "Rows per check result", "Each row of the source carries the first check it fails, in a fixed order."
    )
    st.dataframe(
        audit.rename(
            columns={"check_result": "Check result", "trips": "Rows", "share": "Share"}
        ).style.format({"Rows": "{:,.0f}", "Share": "{:.2%}"}),
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Taxi operations", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    marts = cached_marts(str(marts_dir()))
    fact = marts["fact_hourly"]
    filters = sidebar_filters(fact)
    current = apply_filters(fact, filters)
    prior = apply_filters(fact, previous_period(filters))

    st.markdown(
        '<div class="report-header"><div class="title">New York yellow taxi operations</div>'
        f"<span>{filters.start:%d %b %Y} to {filters.end:%d %b %Y} &middot; "
        f"{len(filters.boroughs)} boroughs &middot; {len(filters.payments)} payment groups</span></div>",
        unsafe_allow_html=True,
    )
    if current.empty:
        st.warning("No trips match the filters.")
        return

    now, before = kpis(current), kpis(prior)
    tiles = [
        ("Trips", "trips", "count"),
        ("Trips per day", "trips_per_day", "plain"),
        ("Revenue", "revenue_usd", "usd_large"),
        ("Mean fare", "mean_fare_usd", "usd"),
        ("Card tip rate", "card_tip_rate", "pct"),
        ("Mean duration", "mean_duration_min", "min"),
    ]
    for column, (label, key, kind) in zip(st.columns(len(tiles)), tiles, strict=True):
        kpi_tile(column, label, now[key], before[key], kind)
    st.write("")

    overview, time_patterns, zones, weather, quality = st.tabs(
        ["Overview", "Time patterns", "Zones", "Weather", "Data quality"]
    )
    with overview:
        overview_tab(current)
    with time_patterns:
        time_tab(current)
    with zones:
        zones_tab(marts, filters)
    with weather:
        weather_tab(marts, filters)
    with quality:
        quality_tab(marts)


main()
