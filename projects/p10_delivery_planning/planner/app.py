"""Delivery planner: plan the routes of one instance, see them on a map, and staff them with drivers.

Run from the project folder: ``make planner``, or
``streamlit run planner/app.py -- --data-dir data``. The data folder needs ``raw/`` from ``make data``.

The shift model runs here with CBC: the planner loads OR-Tools for the routes, and OR-Tools' own HiGHS
library clashes with highspy in one process.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from delivery_planning import shifts as sh
from delivery_planning.config import DATA_DIR, DROP_PENALTY, MAIN_CONFIG, ROUTING_CONFIGS, SCALE
from delivery_planning.data import instance_names, load, schedule
from delivery_planning.report import clock, route_map
from delivery_planning.routing import solve

SHIFT_SECONDS = 20


def data_dir() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("P10_DATA_DIR", DATA_DIR)))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.data_dir


def route_table(instance, routes: list[list[int]]) -> pd.DataFrame:
    rows = []
    for k, route in enumerate(routes, start=1):
        times = schedule(instance, route)
        path = [0, *route, 0]
        rows.append(
            {
                "route": k,
                "stops": len(route),
                "load": int(instance.demand[route].sum()),
                "distance": instance.dist[path[:-1], path[1:]].sum() / SCALE,
                "depart": times["depart"] / SCALE,
                "return": times["ret"] / SCALE,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Delivery planner", layout="wide")
st.title("Delivery planner")
st.caption(
    "Routes with vehicle capacity and time windows on a Solomon benchmark instance (OR-Tools), "
    "then driver shifts for the day from the synthetic pool (Pyomo with CBC)."
)

raw_dir = data_dir() / "raw"
names = instance_names(raw_dir) if raw_dir.exists() else []
if not names:
    st.error(f"No instances in {raw_dir}. Run `make data` first.")
    st.stop()

with st.sidebar:
    st.header("Routes")
    name = st.selectbox("Instance", names, index=names.index("R101") if "R101" in names else 0)
    objective = st.radio(
        "Objective",
        ["distance", "cost_aware"],
        format_func=lambda o: {
            "distance": "Distance (as the best known)",
            "cost_aware": "Distance, vehicles and driver time",
        }[o],
    )
    instance, best = load(raw_dir, name)
    vehicles = st.slider("Vehicles available", 1, instance.vehicles, instance.vehicles)
    seconds = st.slider("Search time (s)", 1, 60, min(ROUTING_CONFIGS[MAIN_CONFIG]["seconds"], 10))
    st.header("Drivers")
    pool = st.slider("Drivers in the pool", 4, 30, 30)
    plan = st.button("Plan", type="primary")

if plan:
    allow_drops = vehicles < instance.vehicles
    result = solve(instance, seconds=seconds, vehicles=vehicles, objective=objective, allow_drops=allow_drops)
    st.session_state["plan"] = {"name": name, "result": result, "pool": pool}

state = st.session_state.get("plan")
if not state or state["name"] != name:
    st.info("Choose an instance and press Plan.")
    st.stop()

result = state["result"]
if not result["found"]:
    st.warning("The search found no solution in the time given. More vehicles or more time may help.")
    st.stop()

distance = result["distance"] / SCALE
cols = st.columns(4)
cols[0].metric(
    "Distance",
    f"{distance:,.1f}",
    f"{distance / best['cost'] - 1:+.2%} against best known",
    delta_color="inverse",
)
cols[1].metric(
    "Vehicles used",
    result["vehicles"],
    f"{result['vehicles'] - len(best['routes']):+d} against best known",
    delta_color="inverse",
)
cols[2].metric("Customers served", f"{result['served']} of {instance.customers}")
cols[3].metric("Search time", f"{result['seconds']:.1f} s")
if result["served"] < instance.customers:
    st.warning(
        f"{instance.customers - result['served']} customers stay unserved: the fleet is too small. "
        f"Each costs {DROP_PENALTY:,.0f} cost units in the objective."
    )

left, right = st.columns([3, 2])
with left:
    page = route_map(instance, result["routes"], f"{name} routes")
    components.html(page.get_root().render(), height=520)
with right:
    st.dataframe(route_table(instance, result["routes"]), hide_index=True, width="stretch")

st.subheader("Driver shifts for the day")
duty_table = sh.duties(0, instance, result["routes"])
drivers = sh.make_drivers(state["pool"])
# Every driver can work this one day; weekly guarantees are left out, so the cost is that of the day.
drivers = drivers.assign(days="1" * 7, guarantee=0.0)
model = sh.build_model(duty_table, drivers)
record = sh.solve_model(model, "cbc", SHIFT_SECONDS)
greedy = sh.greedy(duty_table, drivers)
greedy_cost = sh.roster_cost(greedy, duty_table, drivers)
if "assignment" not in record:
    st.warning(f"The shift model found no roster ({record.get('status')}).")
    st.stop()
cost = sh.roster_cost(record["assignment"], duty_table, drivers)
cols = st.columns(4)
cols[0].metric(
    "Driver cost (shift model)",
    f"{cost['total']:,.0f}",
    f"{cost['total'] - greedy_cost['total']:+,.0f} against greedy",
    delta_color="inverse",
)
cols[1].metric("Drivers on shift", cost["drivers_used"])
cols[2].metric("Agency duties", cost["agency_duties"])
cols[3].metric("Solver status", record["status"])
st.caption(
    "Pay for the hours of this day, without weekly guarantees. Clock times map the depot hours onto "
    "08:00 to 18:00."
)
roster = duty_table.assign(
    driver=duty_table["duty"].map(record["assignment"]).fillna("agency"),
    start=duty_table["start"].map(clock),
    end=duty_table["end"].map(clock),
    hours=duty_table["hours"].round(2),
)
st.dataframe(
    roster[["duty", "driver", "start", "end", "hours", "customers"]], hide_index=True, width="stretch"
)
