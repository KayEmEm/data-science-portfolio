from delivery_planning.data import check_solution
from delivery_planning.routing import cost_coefficients, solve


def test_construction_heuristic_returns_a_feasible_plan(r101):
    instance, _ = r101
    result = solve(instance, metaheuristic=None, seconds=None)
    assert result["found"]
    assert check_solution(instance, result["routes"])["problems"] == []
    assert result["served"] == instance.customers


def test_local_search_does_not_worsen_the_first_solution(r101):
    instance, _ = r101
    first = solve(instance, metaheuristic=None, seconds=None)
    improved = solve(instance, seconds=1)
    assert improved["distance"] <= first["distance"]


def test_a_small_fleet_leaves_customers_unserved_at_a_penalty(r101):
    instance, _ = r101
    result = solve(instance, seconds=1, vehicles=1, allow_drops=True)
    assert result["vehicles"] <= 1
    assert result["served"] < instance.customers
    assert check_solution(instance, result["routes"], require_all=False)["problems"] == []


def test_cost_aware_objective_counts_vehicles(r101):
    instance, _ = r101
    coef = cost_coefficients(instance)
    assert coef["per_vehicle"] > coef["per_tick_span"] > 0
    distance = solve(instance, seconds=1)
    aware = solve(instance, seconds=1, objective="cost_aware")
    assert aware["found"] and aware["served"] == instance.customers
    assert aware["vehicles"] <= distance["vehicles"] + 1
