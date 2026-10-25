"""Tables, figures, route maps and results.json under ``reports/``.

Figures: gap to the best known solution per configuration and instance class, gap against search
time, routes of three instances, solve times of the shift model per solver, the week's roster, and
the cost of the week per fleet cap. Maps: one interactive Folium page per instance in MAP_INSTANCES,
drawn on the plain coordinate grid of the benchmark (the instances have no geographic location).
"""

import argparse
import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import PALETTE, apply_style

from delivery_planning.config import (
    ARTIFACTS_DIR,
    CLASSES,
    DATA_DIR,
    DAY_NAMES,
    MAIN_CONFIG,
    REPORTS_DIR,
    ROUTING_CONFIGS,
    SCALE,
)
from delivery_planning.data import read_instance, read_solution, schedule

MAP_INSTANCES = ["C101", "R101", "RC101"]
ROUTE_COLORS = [
    "#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#c9a227", "#7b2cbf", "#2b9348", "#9d0208",
    "#4361ee", "#bc6c25", "#5e548e", "#e76f51", "#0f4c5c", "#6a994e", "#b5179e", "#495057",
]  # fmt: skip
CONFIG_LABELS = {
    "cheapest_arc": "Cheapest arc, first solution",
    "cheapest_insertion": "Cheapest insertion, first solution",
    "gls_10s": "Guided local search, 10 s",
    "gls_60s": "Guided local search, 60 s",
    "gls_300s": "Guided local search, 300 s",
}


def pct(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def clock(minutes: float) -> str:
    minutes = round(minutes)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


# Figures -------------------------------------------------------------------------------------------


def figure_gaps(summary: pd.DataFrame, configs: list[str], path: Path) -> None:
    classes = [c for c in CLASSES if c in set(summary["class"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(configs)
    for k, config in enumerate(configs):
        rows = summary.loc[summary["config"] == config].set_index("class").reindex(classes)
        x = np.arange(len(classes)) + (k - (len(configs) - 1) / 2) * width
        ax.bar(
            x,
            100 * rows["mean_gap"],
            width * 0.92,
            label=CONFIG_LABELS.get(config, config),
            color=PALETTE[k % 5],
        )
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_ylabel("Mean gap to best known distance (%)")
    ax.set_xlabel("Instance class")
    ax.set_title("Distance above the best known solutions, per instance class")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def figure_time(summary: pd.DataFrame, path: Path) -> None:
    timed = [c for c, spec in ROUTING_CONFIGS.items() if spec["seconds"] and c in set(summary["config"])]
    if len(timed) < 2:
        return
    seconds = [ROUTING_CONFIGS[c]["seconds"] for c in timed]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    classes = [c for c in CLASSES if c in set(summary["class"])]
    for k, cls in enumerate([*classes, "all"]):
        rows = summary.loc[summary["class"] == cls].set_index("config").reindex(timed)
        style = (
            {"color": "black", "linewidth": 2.5}
            if cls == "all"
            else {"color": ROUTE_COLORS[k], "linewidth": 1.5}
        )
        ax.plot(
            seconds,
            100 * rows["mean_gap"],
            marker="o",
            label="all instances" if cls == "all" else cls,
            **style,
        )
    ax.set_xscale("log")
    ax.set_xticks(seconds, [f"{s} s" for s in seconds])
    ax.set_xlabel("Time limit of guided local search (one CPU core)")
    ax.set_ylabel("Mean gap to best known distance (%)")
    ax.set_title("Mean gap to the best known distance by search time")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def draw_routes(ax, instance, routes, title: str) -> None:
    coords = instance.coords
    for k, route in enumerate(routes):
        path = coords[[0, *route, 0]]
        ax.plot(path[:, 0], path[:, 1], color=ROUTE_COLORS[k % len(ROUTE_COLORS)], linewidth=1.2)
    ax.scatter(coords[1:, 0], coords[1:, 1], s=9, color="#495057", zorder=3)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=70, marker="s", color="black", zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_routes(raw_dir: Path, artifacts: Path, config: str, path: Path) -> None:
    names = [n for n in MAP_INSTANCES if (raw_dir / f"{n}.vrp").exists()]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.3))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        instance = read_instance(raw_dir / f"{name}.vrp")
        result = json.loads((artifacts / "benchmark" / config / f"{name}.json").read_text())
        best = read_solution(raw_dir / f"{name}.sol")
        gap = result["distance"] / SCALE / best["cost"] - 1
        distance = result["distance"] / SCALE
        title = f"{name}: {result['vehicles']} routes, {distance:.1f} ({pct(gap, 2)} above best known)"
        draw_routes(ax, instance, result["routes"], title)
    fig.suptitle(
        f"Routes of {CONFIG_LABELS.get(config, config).lower()}; the square is the depot", fontsize=10
    )
    fig.savefig(path)
    plt.close(fig)


def figure_solvers(comparison: pd.DataFrame, path: Path) -> None:
    solved = comparison.loc[comparison["cost"].notna()]
    scenarios = list(dict.fromkeys(comparison["scenario"]))
    solvers = list(dict.fromkeys(comparison["solver"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(solvers)
    for k, solver in enumerate(solvers):
        for j, scenario in enumerate(scenarios):
            x = j + (k - (len(solvers) - 1) / 2) * width
            row = solved.loc[(solved["solver"] == solver) & (solved["scenario"] == scenario)]
            if len(row):
                ax.bar(
                    x,
                    row["seconds"].iloc[0],
                    width * 0.92,
                    color=PALETTE[k],
                    label=solver if j == 0 else None,
                )
                if row["status"].iloc[0] != "optimal":
                    ax.text(x, row["seconds"].iloc[0], "limit", ha="center", va="bottom", fontsize=7)
            else:
                ax.text(x, 0.02, "not run", ha="center", va="bottom", fontsize=7, rotation=90,
                        transform=ax.get_xaxis_transform())  # fmt: skip
    labels = [
        f"{s}\n{int(r.days)} days, {int(r.drivers)} drivers\n{int(r.binaries)} binaries"
        for s, r in comparison.dropna(subset=["binaries"])
        .groupby("scenario", sort=False)
        .first()
        .loc[scenarios]
        .iterrows()
    ]
    ax.set_xticks(range(len(scenarios)), labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Wall time (s, log scale)")
    ax.set_title(
        'Shift model: solve time per solver, one thread each ("limit": stopped at the time limit)',
        fontsize=10,
    )
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def figure_roster(roster: pd.DataFrame, drivers: pd.DataFrame, path: Path) -> None:
    order = [d for d in drivers["driver"] if d in set(roster["driver"])]
    if (roster["driver"] == "agency").any():
        order.append("agency")
    fig, ax = plt.subplots(figsize=(10, 0.22 * len(order) + 1.5))
    rows = {d: i for i, d in enumerate(order)}
    contract = drivers.set_index("driver")["contract"].to_dict()
    for duty in roster.itertuples():
        color = (
            "#9d0208"
            if duty.driver == "agency"
            else PALETTE[0 if contract[duty.driver] == "full_time" else 1]
        )
        left = duty.day * 24 * 60 + duty.start
        ax.barh(rows[duty.driver], duty.end - duty.start, left=left, height=0.7, color=color)
    for day in range(1, 7):
        ax.axvline(day * 24 * 60, color="#adb5bd", linewidth=0.8)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xticks([d * 24 * 60 + 12 * 60 for d in range(7)], DAY_NAMES)
    ax.set_xlim(0, 7 * 24 * 60)
    ax.grid(False)
    ax.set_title(
        "Week roster from the shift model: blue full time, orange part time, red agency (duties may overlap)"
    )
    fig.savefig(path)
    plt.close(fig)


def figure_fleet(fleet: pd.DataFrame, path: Path) -> None:
    parts = [("distance_cost", "distance"), ("vehicle_cost", "vehicles"), ("driver_cost", "drivers"),
             ("unserved_cost", "unserved customers")]  # fmt: skip
    studied = fleet.loc[fleet["objective"] != "benchmark_plan"]
    objectives = list(dict.fromkeys(studied["objective"]))
    fig, axes = plt.subplots(1, len(objectives), figsize=(11, 4.5), sharey=True)
    for ax, objective in zip(np.atleast_1d(axes), objectives, strict=True):
        rows = studied.loc[studied["objective"] == objective].sort_values("cap")
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for k, (column, label) in enumerate(parts):
            values = rows[column].fillna(0).to_numpy()
            color = "#9d0208" if column == "unserved_cost" else PALETTE[k]
            ax.bar(x, values, 0.7, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=1.5)
            bottom += values
        ax.set_xticks(x, rows["cap"].astype(int))
        ax.set_axisbelow(True)
        ax.set_xlabel("Cap on vehicles per day")
        ax.set_title(f"Routing objective: {objective.replace('_', ' ')}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Cost of the week (cost units)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Week of R101 to R107: cost by fleet cap", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


# Maps ----------------------------------------------------------------------------------------------


def route_map(instance, routes: list[list[int]], title: str) -> folium.Map:
    """Routes on the coordinate grid of the instance (Folium simple CRS, no map tiles)."""
    center = [float(instance.coords[:, 1].mean()), float(instance.coords[:, 0].mean())]
    page = folium.Map(location=center, crs="Simple", tiles=None, zoom_start=2, min_zoom=1)
    folium.Rectangle([[0, 0], [100, 100]], color="#dee2e6", weight=1, fill=False).add_to(page)
    for k, route in enumerate(routes):
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
        layer = folium.FeatureGroup(name=f"Route {k + 1} ({len(route)} stops)")
        path = [[float(instance.coords[n, 1]), float(instance.coords[n, 0])] for n in [0, *route, 0]]
        folium.PolyLine(path, color=color, weight=3, opacity=0.85).add_to(layer)
        times = schedule(instance, route)
        for stop, (node, start) in enumerate(zip(route, times["starts"], strict=True), start=1):
            tip = (
                f"Customer {node}, route {k + 1}, stop {stop}<br>demand {instance.demand[node]}<br>"
                f"window {instance.ready[node] / SCALE:g} to {instance.due[node] / SCALE:g}, "
                f"service at {start / SCALE:g}"
            )
            folium.CircleMarker(
                [float(instance.coords[node, 1]), float(instance.coords[node, 0])],
                radius=4, color=color, fill=True, fill_opacity=0.9, tooltip=tip,
            ).add_to(layer)  # fmt: skip
        layer.add_to(page)
    folium.Marker(
        [float(instance.coords[0, 1]), float(instance.coords[0, 0])],
        tooltip=f"Depot, open 0 to {instance.horizon / SCALE:g}",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(page)
    folium.LayerControl(collapsed=True).add_to(page)
    page.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
    return page


# Report --------------------------------------------------------------------------------------------


def benchmark_tables(summary: pd.DataFrame, table: pd.DataFrame, configs: list[str]) -> str:
    overall = summary.loc[summary["class"] == "all"].copy()
    overall = pd.DataFrame(
        {
            "Configuration": [CONFIG_LABELS.get(c, c) for c in overall["config"]],
            "Solved": [
                f"{int(f)} of {int(n)}" for f, n in zip(overall["found"], overall["instances"], strict=True)
            ],
            "Mean gap": overall["mean_gap"].map(pct),
            "Median gap": overall["median_gap"].map(pct),
            "Largest gap": overall["max_gap"].map(pct),
            "Within 1%": overall["within_1pct"].map(lambda v: pct(v, 0)),
            "Vehicles against best known": overall["mean_vehicles_diff"].map(lambda v: f"{v:+.1f}"),
            "Mean seconds": overall["mean_seconds"].map(lambda v: f"{v:.2f}"),
        }
    )
    main = configs[-1] if MAIN_CONFIG not in configs else MAIN_CONFIG
    classes = summary.loc[(summary["config"].isin(configs)) & (summary["class"] != "all")]
    pivot = classes.pivot(index="class", columns="config", values="mean_gap").reindex(columns=configs)
    pivot = pivot.reindex([c for c in CLASSES if c in pivot.index]).map(pct).reset_index()
    pivot.columns = ["Class", *[CONFIG_LABELS.get(c, c) for c in configs]]
    detail = table.loc[
        table["config"] == main, ["instance", "distance", "bks_distance", "gap", "vehicles", "bks_vehicles"]
    ]
    detail = detail.assign(gap=detail["gap"].map(lambda v: pct(v, 2)))
    detail.columns = ["Instance", "Distance", "Best known", "Gap", "Vehicles", "Vehicles in best known"]
    return "\n\n".join(
        [
            "## Routing benchmark\n\nAll instances, each configuration (one CPU core per search).",
            markdown(overall),
            "Mean gap per instance class.",
            markdown(pivot),
            f"### Every instance, {CONFIG_LABELS.get(main, main).lower()}",
            markdown(detail),
        ]
    )


def solver_table(comparison: pd.DataFrame) -> str:
    show = pd.DataFrame(
        {
            "Scenario": comparison["scenario"],
            "Days": comparison["days"],
            "Drivers": comparison["drivers"],
            "Duties": comparison["duties"],
            "Binaries": comparison["binaries"].map(lambda v: "" if pd.isna(v) else f"{int(v):,}"),
            "Constraints": comparison["constraints"].map(lambda v: "" if pd.isna(v) else f"{int(v):,}"),
            "Solver": comparison["solver"],
            "Status": comparison["status"],
            "Cost": comparison["cost"].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}"),
            "Gap to bound": comparison["gap"].map(lambda v: "" if pd.isna(v) else pct(v, 3)),
            "Seconds": comparison["seconds"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
            "Greedy cost": comparison["greedy_cost"].map(lambda v: f"{v:,.1f}"),
            "Saving against greedy": comparison["saving_vs_greedy"].map(
                lambda v: "" if pd.isna(v) else pct(v, 2)
            ),
        }
    )
    notes = comparison.reindex(columns=["scenario", "solver", "status", "message"])
    notes = notes.loc[notes["status"].isin(["error", "skipped"])]
    lines = [f"* {r.scenario}, {r.solver}: {r.message}" for r in notes.itertuples()]
    return "\n\n".join(["## Shift model and solvers", markdown(show), *(["\n".join(lines)] if lines else [])])


def fleet_table(fleet: pd.DataFrame) -> str:
    show = pd.DataFrame(
        {
            "Routing objective": fleet["objective"].str.replace("_", " "),
            "Cap per day": fleet["cap"],
            "Km": fleet["km"].map(lambda v: f"{v:,.0f}"),
            "Vehicle days": fleet["vehicle_days"],
            "Unserved": fleet["unserved"],
            "Duty hours": fleet["duty_hours"].map(lambda v: f"{v:,.1f}"),
            "Distance": fleet["distance_cost"].map(lambda v: f"{v:,.0f}"),
            "Vehicles": fleet["vehicle_cost"].map(lambda v: f"{v:,.0f}"),
            "Drivers": fleet["driver_cost"].map(lambda v: f"{v:,.0f}"),
            "Unserved penalty": fleet["unserved_cost"].map(lambda v: f"{v:,.0f}"),
            "Total": fleet["total_cost"].map(lambda v: f"{v:,.0f}"),
            "MIP status": fleet["mip_status"],
        }
    )
    return "## Fleet size study\n\nCosts in cost units for the week R101 to R107.\n\n" + markdown(show)


def run(data_dir: Path, artifacts: Path, reports: Path, data_label: str) -> dict:
    apply_style()
    ev = artifacts / "evaluation"
    figures = reports / "figures"
    maps = reports / "maps"
    figures.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    table = pd.read_csv(ev / "benchmark.csv")
    summary = pd.read_csv(ev / "benchmark_summary.csv")
    comparison = pd.read_csv(ev / "solver_comparison.csv")
    fleet = pd.read_csv(ev / "fleet.csv")
    roster = pd.read_csv(ev / "roster.csv")
    drivers = pd.read_csv(ev / "drivers.csv", dtype={"days": str})
    configs = list(dict.fromkeys(summary["config"]))
    main = MAIN_CONFIG if MAIN_CONFIG in configs else configs[-1]

    figure_gaps(summary, configs, figures / "gap_by_class.png")
    figure_time(summary, figures / "gap_by_time.png")
    figure_routes(raw_dir, artifacts, main, figures / "routes.png")
    figure_solvers(comparison, figures / "solver_times.png")
    figure_roster(roster, drivers, figures / "roster.png")
    figure_fleet(fleet, figures / "fleet_cost.png")
    for name in MAP_INSTANCES:
        if (raw_dir / f"{name}.vrp").exists():
            instance = read_instance(raw_dir / f"{name}.vrp")
            result = json.loads((artifacts / "benchmark" / main / f"{name}.json").read_text())
            route_map(instance, result["routes"], f"{name} routes").save(str(maps / f"{name}.html"))

    shift_rows = pd.read_csv(ev / "shifts.csv")
    days = range(roster["day"].max() + 1)
    by_day = roster.assign(agency=roster["driver"] == "agency").groupby("day")
    roster_summary = pd.DataFrame(
        {
            "Day": DAY_NAMES[: len(days)],
            "Duties": by_day.size().to_numpy(),
            "Agency duties": by_day["agency"].sum().to_numpy(),
            "Drivers on shift": shift_rows.groupby("day")["driver"]
            .nunique()
            .reindex(days)
            .fillna(0)
            .astype(int)
            .to_numpy(),
            "First start": by_day["start"].min().map(clock).to_numpy(),
            "Last end": by_day["end"].max().map(clock).to_numpy(),
        }
    )
    text = "\n\n".join(
        [
            f"# Project 10 results ({data_label} data)",
            benchmark_tables(summary, table, configs),
            solver_table(comparison),
            "### Week roster of the best full scenario solution\n\n" + markdown(roster_summary),
            fleet_table(fleet),
        ]
    )
    (reports / "results.md").write_text(text + "\n")

    overall = summary.loc[summary["class"] == "all"].set_index("config")
    full = comparison.loc[(comparison["scenario"] == "full") & comparison["cost"].notna()]
    studied = fleet.loc[fleet["objective"] != "benchmark_plan"]
    cheapest = studied.loc[studied["total_cost"].idxmin()]
    reference = fleet.loc[fleet["objective"] == "benchmark_plan"].iloc[0]
    results = {
        "data": data_label,
        "instances": int(overall["instances"].iloc[0]),
        "main_config": main,
        "routing": {
            c: {
                k: float(overall.at[c, k])
                for k in ("mean_gap", "median_gap", "max_gap", "found", "mean_seconds")
            }
            for c in configs
        },
        "shifts_full": {
            r.solver: {"status": r.status, "cost": r.cost, "seconds": r.seconds, "gap": r.gap}
            for r in full.itertuples()
        },
        "greedy_full": float(comparison.loc[comparison["scenario"] == "full", "greedy_cost"].iloc[0]),
        "fleet": {
            "cheapest": {
                "objective": cheapest["objective"],
                "cap": int(cheapest["cap"]),
                "total": float(cheapest["total_cost"]),
            },
            "benchmark_plan_total": float(reference["total_cost"]),
        },
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--data-label", default="Solomon benchmark")
    args = parser.parse_args()
    print(
        json.dumps(run(args.data_dir, args.artifacts, args.reports, args.data_label), indent=2, default=float)
    )


if __name__ == "__main__":
    main()
