"""Driver shifts that cover the routes of a week at minimum cost.

Each route is a duty: it needs one driver from CHECK_MINUTES before departure to CHECK_MINUTES after
return. A driver can take several duties on a day when they do not overlap. A shift runs from the
start of the first duty to the end of the last; it is paid for its span, at least MIN_PAID_HOURS.
Each week a driver is paid at least the contract guarantee, and hours above OVERTIME_AFTER at
OVERTIME_FACTOR times the wage. A duty that no own driver takes goes to an agency driver at
AGENCY_RATE per hour, at least AGENCY_MIN_HOURS.

``build_model`` writes this as a mixed integer model in Pyomo; ``greedy`` is the baseline: duties in
start order, each to the driver with the lowest added cost. ``roster_cost`` and ``check_roster``
price and check any assignment, independent of the model.
"""

import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from delivery_planning.config import (
    AGENCY_MIN_HOURS,
    AGENCY_RATE,
    CHECK_MINUTES,
    CONTRACTS,
    DAY_MINUTES,
    DAY_NAMES,
    DAY_START,
    DAYS_OFF,
    DRIVERS,
    FULL_TIME_SHARE,
    FULL_TIME_WINDOW,
    MAX_SPAN_HOURS,
    MIN_PAID_HOURS,
    MIP_GAP,
    OVERTIME_AFTER,
    OVERTIME_FACTOR,
    PART_TIME_WINDOWS,
    SEED,
)
from delivery_planning.data import Instance, schedule

# Duties and drivers ------------------------------------------------------------------------------


def duties(day: int, instance: Instance, routes: list[list[int]]) -> pd.DataFrame:
    """One duty per route, with clock times in minutes after midnight."""
    minutes_per_tick = DAY_MINUTES / instance.horizon
    rows = []
    for k, route in enumerate(routes):
        times = schedule(instance, route)
        path = [0, *route, 0]
        rows.append(
            {
                "duty": f"{DAY_NAMES[day]}-{k + 1:02d}",
                "day": day,
                "instance": instance.name,
                "start": DAY_START + times["depart"] * minutes_per_tick - CHECK_MINUTES,
                "end": DAY_START + times["ret"] * minutes_per_tick + CHECK_MINUTES,
                "customers": len(route),
                "distance": int(instance.dist[path[:-1], path[1:]].sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["hours"] = (table["end"] - table["start"]) / 60
    return table


def make_drivers(count: int = DRIVERS, days: int = 7, seed: int = SEED) -> pd.DataFrame:
    """The synthetic driver pool: contract, wage, days available and daily window."""
    rng = np.random.default_rng(seed)
    full = round(count * FULL_TIME_SHARE)
    contracts = np.array(["full_time"] * full + ["part_time"] * (count - full))
    rng.shuffle(contracts)
    rows = []
    for i, contract in enumerate(contracts):
        low, high = DAYS_OFF[contract]
        off = set(rng.choice(7, size=int(rng.integers(low, high + 1)), replace=False).tolist())
        if contract == "full_time":
            window_name, window = "day", FULL_TIME_WINDOW
        else:
            window_name = str(rng.choice(list(PART_TIME_WINDOWS)))
            window = PART_TIME_WINDOWS[window_name]
        terms = CONTRACTS[contract]
        rows.append(
            {
                "driver": f"D{i + 1:02d}",
                "contract": contract,
                "wage": terms["wage"],
                "guarantee": terms["guarantee"],
                "max_hours": terms["max_hours"],
                "max_days": terms["max_days"],
                "window": window_name,
                "earliest": window[0] * 60,
                "latest": window[1] * 60,
                "days": "".join("0" if d in off else "1" for d in range(days)),
            }
        )
    return pd.DataFrame(rows)


def allowed(drivers: pd.DataFrame, duty_table: pd.DataFrame) -> list[tuple[str, str]]:
    """(driver, duty) pairs where the driver is available that day and the duty fits the window."""
    pairs = []
    for driver in drivers.itertuples():
        for duty in duty_table.itertuples():
            if (
                driver.days[duty.day] == "1"
                and duty.start >= driver.earliest
                and duty.end <= driver.latest
                and duty.hours <= MAX_SPAN_HOURS
            ):
                pairs.append((driver.driver, duty.duty))
    return pairs


def agency_cost(hours: float) -> float:
    return AGENCY_RATE * max(hours, AGENCY_MIN_HOURS)


# Pricing and checking any assignment ---------------------------------------------------------------


def shifts(assignment: dict[str, str], duty_table: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and day worked: span and paid hours. ``assignment`` maps duty to driver."""
    table = duty_table.set_index("duty")
    rows = []
    groups = defaultdict(list)
    for duty, driver in assignment.items():
        groups[(driver, int(table.at[duty, "day"]))].append(duty)
    for (driver, day), names in sorted(groups.items()):
        start, end = table.loc[names, "start"].min(), table.loc[names, "end"].max()
        span = (end - start) / 60
        rows.append(
            {
                "driver": driver,
                "day": day,
                "duties": len(names),
                "start": start,
                "end": end,
                "span_hours": span,
                "paid_hours": max(span, MIN_PAID_HOURS),
                "driving_hours": float(table.loc[names, "hours"].sum()),
            }
        )
    columns = ["driver", "day", "duties", "start", "end", "span_hours", "paid_hours", "driving_hours"]
    return pd.DataFrame(rows, columns=columns)


def driver_pay(row, hours: float) -> float:
    paid = max(hours, row.guarantee)
    return row.wage * (paid + (OVERTIME_FACTOR - 1) * max(hours - OVERTIME_AFTER, 0.0))


def roster_cost(assignment: dict[str, str], duty_table: pd.DataFrame, drivers: pd.DataFrame) -> dict:
    """Driver pay (with guarantees and overtime) plus agency cost of the duties not assigned."""
    shift_table = shifts(assignment, duty_table)
    hours = shift_table.groupby("driver")["paid_hours"].sum()
    pay = sum(driver_pay(row, float(hours.get(row.driver, 0.0))) for row in drivers.itertuples())
    open_duties = duty_table.loc[~duty_table["duty"].isin(assignment)]
    agency = float(sum(agency_cost(h) for h in open_duties["hours"]))
    guarantee = float((drivers["wage"] * drivers["guarantee"]).sum())
    return {
        "total": pay + agency,
        "driver_pay": pay,
        "agency": agency,
        "guarantee_floor": guarantee,
        "agency_duties": len(open_duties),
        "drivers_used": int(shift_table["driver"].nunique()) if len(shift_table) else 0,
        "shifts": len(shift_table),
        "paid_hours": float(shift_table["paid_hours"].sum()) if len(shift_table) else 0.0,
        "overtime_hours": float(sum(max(h - OVERTIME_AFTER, 0.0) for h in hours)),
    }


def check_roster(assignment: dict[str, str], duty_table: pd.DataFrame, drivers: pd.DataFrame) -> list[str]:
    """Rule violations of an assignment; empty when every rule holds."""
    problems = []
    pairs = set(allowed(drivers, duty_table))
    for duty, driver in assignment.items():
        if (driver, duty) not in pairs:
            problems.append(f"{driver} cannot take {duty}")
    table = duty_table.set_index("duty")
    per_shift = defaultdict(list)
    for duty, driver in assignment.items():
        per_shift[(driver, int(table.at[duty, "day"]))].append(
            (table.at[duty, "start"], table.at[duty, "end"])
        )
    for (driver, day), spans in per_shift.items():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            if start < end - 1e-9:
                problems.append(f"{driver} has overlapping duties on day {day}")
        if (spans[-1][1] - spans[0][0]) / 60 > MAX_SPAN_HOURS + 1e-9:
            problems.append(f"{driver} works longer than {MAX_SPAN_HOURS} hours on day {day}")
    shift_table = shifts(assignment, duty_table)
    for row in drivers.itertuples():
        mine = shift_table.loc[shift_table["driver"] == row.driver]
        if mine["paid_hours"].sum() > row.max_hours + 1e-6:
            problems.append(f"{row.driver} exceeds {row.max_hours} hours")
        if len(mine) > row.max_days:
            problems.append(f"{row.driver} works more than {row.max_days} days")
    return problems


# Greedy baseline -----------------------------------------------------------------------------------


def greedy(duty_table: pd.DataFrame, drivers: pd.DataFrame) -> dict[str, str]:
    """Duties in start order; each goes to the driver with the lowest added cost, or to the agency."""
    pairs = defaultdict(set)
    for driver, duty in allowed(drivers, duty_table):
        pairs[duty].add(driver)
    info = drivers.set_index("driver")
    work = defaultdict(list)  # (driver, day) -> list of (start, end)
    hours = defaultdict(float)
    assignment = {}
    for duty in duty_table.sort_values(["day", "start"]).itertuples():
        best, best_cost = None, agency_cost(duty.hours)
        for driver in sorted(pairs[duty.duty]):
            row = info.loc[driver]
            spans = work[(driver, duty.day)]
            if any(duty.start < end and start < duty.end for start, end in spans):
                continue
            if not spans and sum(1 for (d, _), s in work.items() if d == driver and s) >= row.max_days:
                continue
            old_paid = shift_paid(spans)
            new_paid = shift_paid([*spans, (duty.start, duty.end)])
            if new_paid is None or hours[driver] - old_paid + new_paid > row.max_hours + 1e-9:
                continue
            new_hours = hours[driver] - old_paid + new_paid
            added = driver_pay(row, new_hours) - driver_pay(row, hours[driver])
            if added < best_cost - 1e-9:
                best, best_cost = driver, added
        if best is not None:
            spans = work[(best, duty.day)]
            hours[best] += -shift_paid(spans) + shift_paid([*spans, (duty.start, duty.end)])
            spans.append((duty.start, duty.end))
            assignment[duty.duty] = best
    return assignment


def shift_paid(spans: list[tuple[float, float]]) -> float | None:
    """Paid hours of a shift with these duties; None when the span is too long."""
    if not spans:
        return 0.0
    span = (max(e for _, e in spans) - min(s for s, _ in spans)) / 60
    if span > MAX_SPAN_HOURS + 1e-9:
        return None
    return max(span, MIN_PAID_HOURS)


# Mixed integer model -----------------------------------------------------------------------------


def build_model(duty_table: pd.DataFrame, drivers: pd.DataFrame) -> pyo.ConcreteModel:
    pairs = allowed(drivers, duty_table)
    table = duty_table.set_index("duty")
    info = drivers.set_index("driver")
    day_of = table["day"].astype(int).to_dict()
    shift_keys = sorted({(i, day_of[r]) for i, r in pairs})

    m = pyo.ConcreteModel("driver_shifts")
    m.P = pyo.Set(initialize=pairs, dimen=2)
    m.R = pyo.Set(initialize=list(table.index))
    m.I = pyo.Set(initialize=list(info.index))
    m.K = pyo.Set(initialize=shift_keys, dimen=2)

    m.x = pyo.Var(m.P, within=pyo.Binary)
    m.u = pyo.Var(m.R, within=pyo.Binary)  # duty given to the agency
    m.y = pyo.Var(m.K, within=pyo.Binary)  # driver works that day
    earliest, latest = info["earliest"].astype(float).to_dict(), info["latest"].astype(float).to_dict()
    m.start = pyo.Var(m.K, bounds=lambda m, i, d: (earliest[i], latest[i]))
    m.end = pyo.Var(m.K, bounds=lambda m, i, d: (earliest[i], latest[i]))
    m.paid = pyo.Var(m.K, within=pyo.NonNegativeReals)
    m.hours = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.pay_hours = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.overtime = pyo.Var(m.I, within=pyo.NonNegativeReals)

    by_duty = defaultdict(list)
    by_shift = defaultdict(list)
    for i, r in pairs:
        by_duty[r].append(i)
        by_shift[(i, day_of[r])].append(r)

    m.cover = pyo.Constraint(m.R, rule=lambda m, r: sum(m.x[i, r] for i in by_duty[r]) + m.u[r] == 1)

    # Interval graph: duties that overlap all contain the start of one of them, so one constraint
    # per duty start keeps a driver on at most one duty at a time.
    cliques = []
    for key, names in by_shift.items():
        for t in sorted({table.at[r, "start"] for r in names}):
            members = tuple(r for r in names if table.at[r, "start"] <= t < table.at[r, "end"])
            cliques.append((*key, members))
    m.C = pyo.RangeSet(0, len(cliques) - 1)
    m.one_at_a_time = pyo.Constraint(
        m.C,
        rule=lambda m, c: (
            sum(m.x[cliques[c][0], r] for r in cliques[c][2]) <= m.y[cliques[c][0], cliques[c][1]]
        ),
    )
    # Big M terms as small as the driver's window allows: without the duty, the shift start can be
    # anywhere up to the end of the window, and the shift end anywhere from its start.
    m.starts_before = pyo.Constraint(
        m.P,
        rule=lambda m, i, r: (
            m.start[i, day_of[r]]
            <= table.at[r, "start"] + (latest[i] - table.at[r, "start"]) * (1 - m.x[i, r])
        ),
    )
    m.ends_after = pyo.Constraint(
        m.P,
        rule=lambda m, i, r: (
            m.end[i, day_of[r]] >= table.at[r, "end"] - (table.at[r, "end"] - earliest[i]) * (1 - m.x[i, r])
        ),
    )
    m.span = pyo.Constraint(
        m.K, rule=lambda m, i, d: m.end[i, d] - m.start[i, d] <= MAX_SPAN_HOURS * 60 * m.y[i, d]
    )
    m.paid_span = pyo.Constraint(m.K, rule=lambda m, i, d: m.paid[i, d] >= (m.end[i, d] - m.start[i, d]) / 60)
    m.paid_minimum = pyo.Constraint(m.K, rule=lambda m, i, d: m.paid[i, d] >= MIN_PAID_HOURS * m.y[i, d])
    # Valid inequality: duties of one shift do not overlap, so the span is at least their total length.
    # It is implied by the constraints above for integer solutions and tightens the relaxation.
    m.paid_duties = pyo.Constraint(
        m.K,
        rule=lambda m, i, d: m.paid[i, d] >= sum(table.at[r, "hours"] * m.x[i, r] for r in by_shift[(i, d)]),
    )

    days_of = defaultdict(list)
    for i, d in shift_keys:
        days_of[i].append(d)
    m.week_hours = pyo.Constraint(m.I, rule=lambda m, i: m.hours[i] == sum(m.paid[i, d] for d in days_of[i]))
    m.max_hours = pyo.Constraint(m.I, rule=lambda m, i: m.hours[i] <= float(info.at[i, "max_hours"]))
    m.max_days = pyo.Constraint(
        m.I,
        rule=lambda m, i: (
            sum(m.y[i, d] for d in days_of[i]) <= int(info.at[i, "max_days"])
            if days_of[i]
            else pyo.Constraint.Skip
        ),
    )
    m.guarantee = pyo.Constraint(m.I, rule=lambda m, i: m.pay_hours[i] >= float(info.at[i, "guarantee"]))
    m.worked = pyo.Constraint(m.I, rule=lambda m, i: m.pay_hours[i] >= m.hours[i])
    m.over = pyo.Constraint(m.I, rule=lambda m, i: m.overtime[i] >= m.hours[i] - OVERTIME_AFTER)

    hours = table["hours"].to_dict()
    m.cost = pyo.Objective(
        expr=sum(
            float(info.at[i, "wage"]) * (m.pay_hours[i] + (OVERTIME_FACTOR - 1) * m.overtime[i]) for i in m.I
        )
        + sum(agency_cost(hours[r]) * m.u[r] for r in m.R),
        sense=pyo.minimize,
    )
    return m


def model_size(m: pyo.ConcreteModel) -> dict:
    variables = list(m.component_data_objects(pyo.Var, active=True))
    return {
        "variables": len(variables),
        "binaries": sum(1 for v in variables if v.is_binary()),
        "constraints": sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True)),
    }


def solver(name: str, seconds: float, gap: float = MIP_GAP):
    """A Pyomo solver with a time limit, a relative gap and one thread."""
    if name == "highs":
        if "ortools" in sys.modules:
            # OR-Tools ships its own HiGHS build; with both loaded, highspy fails on a missing symbol.
            raise RuntimeError("HiGHS through highspy cannot run in a process that has loaded OR-Tools")
        opt = pyo.SolverFactory("appsi_highs")
        opt.config.mip_gap = gap
        opt.highs_options = {"threads": 1, "random_seed": SEED}
        return opt
    if name == "cbc":
        import cbcbox

        opt = pyo.SolverFactory("cbc", executable=cbcbox.cbc_bin_path())
        opt.options.update({"seconds": seconds, "ratioGap": gap, "threads": 1, "randomSeed": SEED})
        return opt
    if name == "gurobi":
        opt = pyo.SolverFactory("gurobi_direct")
        opt.options.update({"TimeLimit": seconds, "MIPGap": gap, "Threads": 1, "Seed": SEED})
        return opt
    raise ValueError(f"unknown solver {name}")


def cbc_bound(log: str) -> float | None:
    """The lower bound that CBC prints at the end of its log."""
    found = re.findall(r"^Lower bound:\s+(\S+)", log, flags=re.MULTILINE)
    return float(found[-1]) if found else None


def solve_model(m: pyo.ConcreteModel, name: str, seconds: float, gap: float = MIP_GAP) -> dict:
    """Solve, and return the record: status, objective, best bound, gap, wall time, assignment.

    The objective is the solver's; at a solution that is not optimal it can exceed the priced cost of
    the same roster (``roster_cost``), because the pay variables need not be at their lower bounds.
    """
    started = time.perf_counter()
    opt = solver(name, seconds, gap)
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "solver.log"
        try:
            # Only CBC needs its log, for the lower bound. The HiGHS interface takes no log file, and it
            # resets its time limit on each call unless the limit is passed here.
            extra = {"cbc": {"logfile": str(log)}, "highs": {"timelimit": seconds}}.get(name, {})
            results = opt.solve(m, load_solutions=False, **extra)
        except Exception as error:  # license limits and missing binaries end up here
            return {"solver": name, "status": "error", "message": str(error).splitlines()[0][:200]}
        text = log.read_text() if log.exists() else ""
    elapsed = time.perf_counter() - started
    term = str(results.solver.termination_condition).split(".")[-1]
    record = {"solver": name, "seconds": round(elapsed, 2), "status": term}
    found = len(results.solution) > 0
    if not found:
        return record
    m.solutions.load_from(results)
    objective = pyo.value(m.cost)
    if term == "optimal":
        bound = objective
    elif name == "cbc":
        bound = cbc_bound(text)
    else:
        lower = results.problem.lower_bound
        bound = float(lower) if lower is not None and np.isfinite(float(lower)) else None
    record.update(
        objective=objective,
        bound=bound,
        gap=(objective - bound) / abs(objective) if bound is not None and objective else None,
        assignment={r: i for (i, r) in m.P if pyo.value(m.x[i, r]) > 0.5},
    )
    return record
