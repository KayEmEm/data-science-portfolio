"""Benchmark tables, the driver shift model with three solvers, and the fleet size study.

Reads the searches of ``train.py`` and writes under ``artifacts/evaluation/``:

* ``benchmark.csv`` and ``benchmark_summary.csv``: distance and gap to the best known solution per
  instance and configuration, and the summary per configuration and instance class;
* ``duties.csv`` and ``drivers.csv``: the week's duties from the main configuration and the
  synthetic driver pool;
* ``mip/<scenario>_<solver>.json`` and ``solver_comparison.csv``: the shift model for each scenario
  with HiGHS, CBC and Gurobi, next to the greedy baseline; ``roster.csv`` and ``shifts.csv`` from the
  best solution of the full scenario;
* ``fleet/<objective>_cap<k>.json`` and ``fleet.csv``: for each cap on vehicles per day and each
  routing objective, the week's distance, vehicles and unserved customers, the driver cost from the
  shift model (HiGHS), and the total cost.

Each MIP result is stored and skipped when run again (``--force`` solves again).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from delivery_planning import shifts as sh
from delivery_planning.config import (
    ARTIFACTS_DIR,
    CLASSES,
    COST_PER_KM,
    COST_PER_VEHICLE_DAY,
    DATA_DIR,
    DRIVERS,
    DROP_PENALTY,
    MAIN_CONFIG,
    MIP_SECONDS,
    OBJECTIVES,
    QUICK,
    SCALE,
    SCENARIOS,
    SOLVERS,
    WEEK,
    WORKERS,
)
from delivery_planning.data import read_instance
from delivery_planning.train import settings

EXPERIMENT = "p10_delivery_planning"


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def store(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(record))
    partial.rename(path)


# Routing benchmark -------------------------------------------------------------------------------


def benchmark(data_dir: Path, artifacts: Path, configs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    instances = pd.read_csv(data_dir / "instances.csv")
    rows = []
    for config in configs:
        for item in instances.to_dict("records"):
            result = read_json(artifacts / "benchmark" / config / f"{item['instance']}.json")
            distance = result["distance"] / SCALE if result["found"] else np.nan
            rows.append(
                {
                    "config": config,
                    "instance": item["instance"],
                    "class": item["class"],
                    "found": result["found"],
                    "distance": distance,
                    "bks_distance": item["bks_distance"],
                    "gap": distance / item["bks_distance"] - 1 if result["found"] else np.nan,
                    "vehicles": result["vehicles"] if result["found"] else np.nan,
                    "bks_vehicles": item["bks_vehicles"],
                    "seconds": result["seconds"],
                }
            )
    table = pd.DataFrame(rows)

    def summary(frame: pd.DataFrame) -> pd.Series:
        found = frame.loc[frame["found"]]
        return pd.Series(
            {
                "instances": len(frame),
                "found": int(frame["found"].sum()),
                "mean_gap": found["gap"].mean(),
                "median_gap": found["gap"].median(),
                "max_gap": found["gap"].max(),
                "within_1pct": (found["gap"] <= 0.01).mean() if len(found) else np.nan,
                "at_bks": (found["gap"] <= 1e-9).mean() if len(found) else np.nan,
                "mean_vehicles_diff": (found["vehicles"] - found["bks_vehicles"]).mean(),
                "mean_seconds": frame["seconds"].mean(),
            }
        )

    parts = []
    for config, frame in table.groupby("config", sort=False):
        per_class = [
            summary(frame.loc[frame["class"] == c]).rename(c) for c in CLASSES if (frame["class"] == c).any()
        ]
        block = pd.DataFrame([*per_class, summary(frame).rename("all")])
        block.index.name = "class"
        parts.append(block.reset_index().assign(config=config))
    return table, pd.concat(parts, ignore_index=True)


# Week duties ---------------------------------------------------------------------------------------


def week_duties(raw_dir: Path, results: list[dict]) -> pd.DataFrame:
    frames = []
    for day, (name, result) in enumerate(zip(WEEK, results, strict=True)):
        frames.append(sh.duties(day, read_instance(raw_dir / f"{name}.vrp"), result["routes"]))
    return pd.concat(frames, ignore_index=True)


def solve_scenario(duty_table: pd.DataFrame, drivers: pd.DataFrame, solver: str, seconds: float) -> dict:
    model = sh.build_model(duty_table, drivers)
    size = sh.model_size(model)
    record = {**size, **sh.solve_model(model, solver, seconds)}
    if "assignment" in record:
        assignment = record["assignment"]
        problems = sh.check_roster(assignment, duty_table, drivers)
        if problems:
            raise RuntimeError(f"{solver}: the solution breaks a rule: {problems[:3]}")
        priced = sh.roster_cost(assignment, duty_table, drivers)["total"]
        if priced > record["objective"] + 1e-3:
            raise RuntimeError(
                f"{solver}: the priced roster {priced} costs more than the objective {record['objective']}"
            )
        record["cost"] = priced
    return record


def solver_available(name: str) -> str | None:
    """None when the solver can run, else the reason it cannot."""
    if name == "gurobi":
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            return "not installed (optional extra gurobi)"
    return None


def solver_comparison(
    duty_table: pd.DataFrame,
    drivers: pd.DataFrame,
    out_dir: Path,
    scenarios: dict,
    seconds: float,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    rows, best_full = [], None
    for scenario, spec in scenarios.items():
        part = duty_table.loc[duty_table["day"] < spec["days"]].reset_index(drop=True)
        pool = drivers.head(spec["drivers"])
        greedy = sh.greedy(part, pool)
        if sh.check_roster(greedy, part, pool):
            raise RuntimeError("the greedy roster breaks a rule")
        greedy_cost = sh.roster_cost(greedy, part, pool)
        base = {
            "scenario": scenario,
            "days": spec["days"],
            "drivers": spec["drivers"],
            "duties": len(part),
            "greedy_cost": greedy_cost["total"],
            "greedy_agency_duties": greedy_cost["agency_duties"],
        }
        for solver in SOLVERS:
            path = out_dir / "mip" / f"{scenario}_{solver}.json"
            if path.exists() and not force:
                record = read_json(path)
            else:
                reason = solver_available(solver)
                record = (
                    {"solver": solver, "status": "skipped", "message": reason}
                    if reason
                    else solve_scenario(part, pool, solver, seconds)
                )
                store(path, record)
            assignment = record.pop("assignment", None)
            if assignment is not None:
                cost = sh.roster_cost(assignment, part, pool)
                record.update(
                    agency_duties=cost["agency_duties"],
                    drivers_used=cost["drivers_used"],
                    paid_hours=cost["paid_hours"],
                    saving_vs_greedy=1 - record["cost"] / greedy_cost["total"],
                )
                if scenario == "full" and (best_full is None or record["cost"] < best_full[0]):
                    best_full = (record["cost"], assignment, part, pool)
            rows.append({**base, **record})
    return pd.DataFrame(rows), best_full


# Fleet size study ----------------------------------------------------------------------------------


def fleet_job(
    raw_dir: str, artifacts: str, objective: str, cap: int, seconds: float, drivers_count: int
) -> dict:
    results = [read_json(Path(artifacts) / "fleet" / objective / f"cap{cap}" / f"{n}.json") for n in WEEK]
    return week_cost(Path(raw_dir), results, objective, cap, seconds, drivers_count)


def week_cost(
    raw_dir: Path, results: list[dict], objective: str, cap: int, seconds: float, drivers_count: int
) -> dict:
    """Routing and driver cost of one week plan; the driver cost from the shift model with HiGHS."""
    duty_table = week_duties(raw_dir, results)
    drivers = sh.make_drivers(drivers_count)
    record = solve_scenario(duty_table, drivers, "highs", seconds)
    assignment = record.pop("assignment", None)
    customers = sum(read_instance(raw_dir / f"{n}.vrp").customers for n in WEEK)
    km = sum(r["distance"] for r in results) / SCALE
    vehicle_days = sum(r["vehicles"] for r in results)
    unserved = customers - sum(r["served"] for r in results)
    driver_cost = record.get("cost", np.nan)
    agency = sh.roster_cost(assignment, duty_table, drivers)["agency_duties"] if assignment else None
    return {
        "objective": objective,
        "cap": cap,
        "km": km,
        "vehicle_days": vehicle_days,
        "max_vehicles": max(r["vehicles"] for r in results),
        "unserved": unserved,
        "duty_hours": float(duty_table["hours"].sum()),
        "distance_cost": COST_PER_KM * km,
        "vehicle_cost": COST_PER_VEHICLE_DAY * vehicle_days,
        "driver_cost": driver_cost,
        "unserved_cost": DROP_PENALTY * unserved,
        "mip_status": record.get("status"),
        "mip_gap": record.get("gap"),
        "agency_duties": agency,
    }


def fleet_study(
    data_dir: Path,
    artifacts: Path,
    out_dir: Path,
    caps: list[int],
    seconds: float,
    drivers_count: int,
    force: bool,
) -> pd.DataFrame:
    todo = [
        (o, c)
        for o in OBJECTIVES
        for c in caps
        if force or not (out_dir / "fleet" / f"{o}_cap{c}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            (o, c): pool.submit(
                fleet_job, str(data_dir / "raw"), str(artifacts), o, c, seconds, drivers_count
            )
            for o, c in todo
        }
        for (o, c), future in futures.items():
            store(out_dir / "fleet" / f"{o}_cap{c}.json", future.result())
            print(f"  fleet {o} cap {c}: stored", flush=True)
    table = pd.DataFrame(
        [read_json(out_dir / "fleet" / f"{o}_cap{c}.json") for o in OBJECTIVES for c in caps]
    )
    table["total_cost"] = table[["distance_cost", "vehicle_cost", "driver_cost", "unserved_cost"]].sum(axis=1)
    return table


# Main ----------------------------------------------------------------------------------------------


def run(data_dir: Path, artifacts: Path, quick: bool, force: bool) -> None:
    out_dir = artifacts / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = settings(quick)
    configs = list(cfg["configs"])
    main_config = MAIN_CONFIG if MAIN_CONFIG in configs else configs[-1]
    seconds = QUICK["mip_seconds"] if quick else MIP_SECONDS
    drivers_count = QUICK["drivers"] if quick else DRIVERS
    scenarios = (
        {"small": {"days": 2, "drivers": 6}, "full": {"days": 7, "drivers": drivers_count}}
        if quick
        else SCENARIOS
    )
    setup_mlflow(artifacts)

    table, summary = benchmark(data_dir, artifacts, configs)
    table.to_csv(out_dir / "benchmark.csv", index=False)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    for config in configs:
        row = summary.loc[(summary["config"] == config) & (summary["class"] == "all")].iloc[0]
        with mlflow.start_run(run_name=f"routing_{config}"):
            mlflow.log_params({"stage": "routing", "config": config, **cfg["configs"][config]})
            mlflow.log_metrics({k: float(row[k]) for k in ("mean_gap", "max_gap", "found", "mean_seconds")})
    print(summary.loc[summary["class"] == "all"].to_string(index=False))

    main_results = [read_json(artifacts / "benchmark" / main_config / f"{n}.json") for n in WEEK]
    duty_table = week_duties(data_dir / "raw", main_results)
    drivers = sh.make_drivers(drivers_count)
    duty_table.to_csv(out_dir / "duties.csv", index=False)
    drivers.to_csv(out_dir / "drivers.csv", index=False)

    comparison, best_full = solver_comparison(duty_table, drivers, out_dir, scenarios, seconds, force)
    comparison.to_csv(out_dir / "solver_comparison.csv", index=False)
    for row in comparison.itertuples():
        with mlflow.start_run(run_name=f"shifts_{row.scenario}_{row.solver}"):
            mlflow.log_params(
                {"stage": "shifts", "scenario": row.scenario, "solver": row.solver, "status": row.status}
            )
            metrics = {k: getattr(row, k, np.nan) for k in ("cost", "bound", "gap", "seconds", "greedy_cost")}
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
    shown = [
        c
        for c in ("scenario", "solver", "status", "cost", "bound", "seconds", "greedy_cost")
        if c in comparison
    ]
    print(comparison[shown].to_string())
    if best_full is not None:
        _, assignment, part, pool = best_full
        roster = part.assign(driver=part["duty"].map(assignment).fillna("agency"))
        roster.to_csv(out_dir / "roster.csv", index=False)
        sh.shifts(assignment, part).to_csv(out_dir / "shifts.csv", index=False)

    fleet = fleet_study(data_dir, artifacts, out_dir, cfg["caps"], seconds, drivers_count, force)
    reference = week_cost(data_dir / "raw", main_results, "benchmark_plan", 25, seconds, drivers_count)
    reference["total_cost"] = sum(
        reference[k] for k in ("distance_cost", "vehicle_cost", "driver_cost", "unserved_cost")
    )
    fleet = pd.concat([fleet, pd.DataFrame([reference])], ignore_index=True)
    fleet.to_csv(out_dir / "fleet.csv", index=False)
    for row in fleet.itertuples():
        with mlflow.start_run(run_name=f"fleet_{row.objective}_cap{row.cap}"):
            mlflow.log_params({"stage": "fleet", "objective": row.objective, "cap": row.cap})
            mlflow.log_metrics({"total_cost": row.total_cost, "km": row.km, "unserved": row.unserved})
    print(
        fleet[["objective", "cap", "km", "vehicle_days", "unserved", "driver_cost", "total_cost"]].to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--quick", action="store_true", help="the quick settings of train.py, short MIP limits"
    )
    parser.add_argument("--force", action="store_true", help="solve stored MIPs again")
    args = parse_args(parser, __file__)
    run(args.data_dir, args.artifacts, args.quick, args.force)


if __name__ == "__main__":
    main()
