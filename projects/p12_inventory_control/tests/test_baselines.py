from conftest import constant_problem

from inventory_rl.baselines import tune
from inventory_rl.dynamics import s_s, seeded_scenarios, simulate


def test_tuned_s_s_beats_or_matches_base_stock(problem):
    result, grids = tune(problem, 30)
    assert result["s_S"]["tune_cost"] <= result["base_stock"]["tune_cost"] + 1e-9
    assert 0 <= result["s_S"]["s"] < result["s_S"]["S"] <= problem.capacity
    assert grids["s_S"]["cost"].min() == result["s_S"]["tune_cost"]
    # The tuned point is the best of its grid on the tuning scenarios.
    scenarios = seeded_scenarios(problem, 30_000, 30)
    tuned = simulate(problem, scenarios, s_s(result["s_S"]["s"], result["s_S"]["S"]))["totals"]["cost"].mean()
    assert abs(tuned - result["s_S"]["tune_cost"]) < 1e-6


def test_free_orders_make_daily_review_best():
    problem = constant_problem(6, 3, order_cost=0.0, dispersion=2.0, horizon=60)
    result, _ = tune(problem, 20)
    assert result["base_stock"]["R"] == 1
