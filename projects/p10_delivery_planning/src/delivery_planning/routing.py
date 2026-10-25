"""Vehicle routing with capacity and time windows in OR-Tools.

Two objectives:

* ``distance``: total distance, the objective of the DIMACS best known solutions;
* ``cost_aware``: distance cost, a fixed cost per vehicle used, and driver time, charged per minute
  from departure to return (the span of each route). It plans for the cost of the whole operation.

With ``allow_drops`` a customer can stay unserved at ``DROP_PENALTY`` in the objective; the fleet
size study needs it when the cap on vehicles is too small to serve every customer.
"""

import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from delivery_planning.config import (
    CHECK_MINUTES,
    CONTRACTS,
    COST_PER_KM,
    COST_PER_VEHICLE_DAY,
    DAY_MINUTES,
    DROP_PENALTY,
    SCALE,
)
from delivery_planning.data import Instance, check_solution

MILLI = 1000  # the cost aware objective is in thousandths of a cost unit, as OR-Tools needs integers
FIRST_SOLUTION_LIMIT = 60  # seconds for a construction heuristic without local search


def cost_coefficients(instance: Instance) -> dict:
    """Integer coefficients of the cost aware objective, in thousandths of a cost unit."""
    wage = CONTRACTS["full_time"]["wage"]
    minutes_per_tick = DAY_MINUTES / instance.horizon  # one tick is a tenth of a time unit
    return {
        "per_tick_distance": round(COST_PER_KM / SCALE * MILLI),
        "per_vehicle": round((COST_PER_VEHICLE_DAY + 2 * CHECK_MINUTES / 60 * wage) * MILLI),
        "per_tick_span": round(wage / 60 * minutes_per_tick * MILLI),
        "drop": round(DROP_PENALTY * MILLI),
    }


def solve(
    instance: Instance,
    first: str = "PARALLEL_CHEAPEST_INSERTION",
    metaheuristic: str | None = "GUIDED_LOCAL_SEARCH",
    seconds: float | None = 10,
    vehicles: int | None = None,
    objective: str = "distance",
    allow_drops: bool = False,
) -> dict:
    """Routes (lists of customer indices), distance in tenths, and the search record."""
    fleet = vehicles or instance.vehicles
    n = instance.customers + 1
    manager = pywrapcp.RoutingIndexManager(n, fleet, 0)
    routing = pywrapcp.RoutingModel(manager)
    dist = instance.dist.tolist()
    service = instance.service.tolist()
    demand = instance.demand.tolist()

    def node(index):
        return manager.IndexToNode(index)

    if objective == "distance":
        arc = routing.RegisterTransitCallback(lambda i, j: dist[node(i)][node(j)])
        drop = round(DROP_PENALTY / (COST_PER_KM / SCALE))
    elif objective == "cost_aware":
        coef = cost_coefficients(instance)
        per_tick = coef["per_tick_distance"]
        arc = routing.RegisterTransitCallback(lambda i, j: per_tick * dist[node(i)][node(j)])
        routing.SetFixedCostOfAllVehicles(coef["per_vehicle"])
        drop = coef["drop"]
    else:
        raise ValueError(f"unknown objective {objective}")
    routing.SetArcCostEvaluatorOfAllVehicles(arc)

    load = routing.RegisterUnaryTransitCallback(lambda i: demand[node(i)])
    routing.AddDimensionWithVehicleCapacity(load, 0, [instance.capacity] * fleet, True, "Capacity")
    travel = routing.RegisterTransitCallback(lambda i, j: service[node(i)] + dist[node(i)][node(j)])
    horizon = instance.horizon
    routing.AddDimension(travel, horizon, horizon, False, "Time")
    clock = routing.GetDimensionOrDie("Time")
    for customer in range(1, n):
        index = manager.NodeToIndex(customer)
        clock.CumulVar(index).SetRange(int(instance.ready[customer]), int(instance.due[customer]))
        if allow_drops:
            routing.AddDisjunction([index], drop)
    for v in range(fleet):
        for index in (routing.Start(v), routing.End(v)):
            clock.CumulVar(index).SetRange(int(instance.ready[0]), horizon)
        if objective == "cost_aware":
            # The departure is otherwise free; the span cost pulls it as late as the route allows.
            routing.AddVariableMaximizedByFinalizer(clock.CumulVar(routing.Start(v)))
            routing.AddVariableMinimizedByFinalizer(clock.CumulVar(routing.End(v)))
    if objective == "cost_aware":
        clock.SetSpanCostCoefficientForAllVehicles(coef["per_tick_span"])

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, first)
    if metaheuristic:
        params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
        params.time_limit.FromMilliseconds(round(seconds * 1000))
    else:
        params.solution_limit = 1
        params.time_limit.FromSeconds(FIRST_SOLUTION_LIMIT)

    started = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - started
    record = {
        "instance": instance.name,
        "objective": objective,
        "fleet": fleet,
        "seconds": round(elapsed, 3),
        "status": int(routing.status()),
    }
    if solution is None:
        return {**record, "found": False, "routes": [], "distance": None, "vehicles": 0, "served": 0}
    routes = []
    for v in range(fleet):
        index, route = routing.Start(v), []
        index = solution.Value(routing.NextVar(index))
        while not routing.IsEnd(index):
            route.append(node(index))
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    check = check_solution(instance, routes, require_all=not allow_drops)
    if check["problems"]:
        raise RuntimeError(
            f"{instance.name}: OR-Tools returned a solution that breaks a rule: {check['problems']}"
        )
    return {
        **record,
        "found": True,
        "routes": routes,
        "distance": check["distance"],
        "vehicles": check["vehicles"],
        "served": check["served"],
        "objective_value": int(solution.ObjectiveValue()),
    }
