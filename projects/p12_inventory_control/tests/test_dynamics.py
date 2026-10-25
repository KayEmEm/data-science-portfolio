import numpy as np
import pytest
from conftest import constant_problem

from inventory_rl.dynamics import (
    Batch,
    Scenarios,
    action_to_order,
    base_stock,
    order_to_action,
    s_s,
    seeded_scenarios,
    simulate,
)


def fixed_scenarios(problem, units, n=1, on_hand=None):
    horizon = problem.horizon
    return Scenarios(
        np.full((n, horizon), units, dtype="int64"),
        np.full((n, horizon), problem.lead_times[0], dtype="int64"),
        np.ones(n, dtype="int64"),
        np.full(n, problem.initial_stock() if on_hand is None else on_hand, dtype="int64"),
    )


def test_stock_balance_and_cost_parts(problem):
    scenarios = seeded_scenarios(problem, 1, 20)
    batch = Batch(problem, scenarios)
    rng = np.random.default_rng(0)
    while not batch.done:
        start, position = batch.on_hand.copy(), batch.position.copy()
        out = batch.step(rng.integers(0, 40, size=batch.size))
        assert (out["sales"] + out["lost"] == out["demand"]).all()
        assert (out["on_hand_end"] == start - out["sales"]).all()
        assert (batch.on_hand == out["on_hand_end"] + out["arrived"]).all()
        assert (batch.position == position + out["order"] - out["sales"]).all()
        assert (position + out["order"] <= problem.capacity).all()
        parts = out["order_cost"] + out["holding_cost"] + out["lost_cost"]
        np.testing.assert_allclose(out["cost"], parts)
        assert (batch.pipeline >= 0).all() and (batch.on_hand >= 0).all()


def test_an_order_arrives_after_its_lead_time():
    problem = constant_problem(0, 4)
    batch = Batch(problem, fixed_scenarios(problem, 0, on_hand=0))
    arrived = [batch.step(np.array([9 if day == 0 else 0]))["arrived"][0] for day in range(6)]
    # Placed on day 0 with a lead time of 4: on hand on the morning of day 4.
    assert arrived == [0, 0, 0, 9, 0, 0]
    assert batch.on_hand[0] == 9


def test_reorder_point_that_covers_the_lead_time_loses_nothing():
    units, lead = 5, 3
    problem = constant_problem(units, lead)
    scenarios = fixed_scenarios(problem, units, on_hand=40)
    covered = simulate(problem, scenarios, s_s(units * lead, 60))["totals"]
    assert covered["lost"][0] == 0
    short = simulate(problem, scenarios, s_s(units * (lead - 1), 60))["totals"]
    assert short["lost"][0] > 0
    assert short["cost"][0] > covered["cost"][0]


def test_base_stock_orders_only_on_review_days():
    problem = constant_problem(5, 2)
    trace = simulate(problem, fixed_scenarios(problem, 5), base_stock(50, 7), trace=1)["trace"]
    days = np.flatnonzero(trace["order"][0] > 0)
    assert len(days) and (days % 7 == 0).all()


def test_policies_accept_one_parameter_per_episode(problem):
    scenarios = seeded_scenarios(problem, 3, 4)
    together = simulate(problem, scenarios, s_s(np.array([5, 20, 40, 60]), np.array([80, 90, 100, 150])))
    for i, (low, high) in enumerate([(5, 80), (20, 90), (40, 100), (60, 150)]):
        alone = simulate(problem, scenarios.take(i), s_s(low, high))
        assert together["totals"]["cost"][i] == pytest.approx(alone["totals"]["cost"][0])


def test_seeded_scenarios_repeat(problem):
    a, b = seeded_scenarios(problem, 7, 5), seeded_scenarios(problem, 7, 5)
    assert (a.demand == b.demand).all() and (a.lead == b.lead).all()
    assert not (seeded_scenarios(problem, 8, 5).demand == a.demand).all()
    assert set(np.unique(a.lead)) <= set(problem.lead_times)


def test_replay_uses_the_given_path(problem):
    path = np.arange(problem.horizon)
    scenarios = seeded_scenarios(problem, 1, 3, demand=path)
    assert (scenarios.demand == path).all()


def test_action_and_order_map_into_each_other():
    position = np.array([0, 10, 50, 99])
    order = np.array([30, 0, 25, 1])
    action = order_to_action(order, position, 100)
    assert (action_to_order(action, position, 100) == order).all()
    assert (action_to_order(np.array([-1.0, -1.0, -1.0, -1.0]), position, 100) == 0).all()
    assert (action_to_order(np.ones(4), position, 100) == 100 - position).all()


def test_a_finished_batch_refuses_more_days():
    problem = constant_problem(1, 2, horizon=3)
    batch = Batch(problem, fixed_scenarios(problem, 1))
    for _ in range(3):
        batch.step(np.zeros(1))
    with pytest.raises(RuntimeError):
        batch.step(np.zeros(1))
