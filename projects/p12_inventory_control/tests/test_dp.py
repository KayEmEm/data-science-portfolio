from dataclasses import replace

import numpy as np
import pytest

from inventory_rl.dp import DP_LEAD, dp_policy, solve
from inventory_rl.dynamics import Problem, s_s, seeded_scenarios, simulate


@pytest.fixture
def small():
    return Problem(
        weekday_mean=(1.2, 1.0, 0.8, 0.6, 0.6, 0.8, 1.0),
        dispersion=3.0,
        holding=0.02,
        lost_sale=2.0,
        order_cost=4.0,
        capacity=14,
        lead_times=(DP_LEAD,),
        lead_probs=(1.0,),
        horizon=56,
        name="small",
    )


def test_simulator_reproduces_the_expected_optimum(small):
    policy, value = solve(small, first_wday=1)
    start = small.initial_stock()
    expected = value[(start,) + (0,) * (DP_LEAD - 1)]
    cost = simulate(small, seeded_scenarios(small, 0, 4000, first_wday=1), dp_policy(policy))["totals"][
        "cost"
    ]
    half = 3 * cost.std(ddof=1) / np.sqrt(len(cost))
    assert abs(cost.mean() - expected) < half


def test_no_simple_policy_beats_the_optimum(small):
    policy, _ = solve(small, first_wday=1)
    scenarios = seeded_scenarios(small, 1, 2000, first_wday=1)
    optimal = simulate(small, scenarios, dp_policy(policy))["totals"]["cost"].mean()
    for low, high in [(1, 8), (2, 10), (3, 14), (0, 6)]:
        assert simulate(small, scenarios, s_s(low, high))["totals"]["cost"].mean() >= optimal - 0.5


def test_dynamic_program_needs_the_fixed_lead_time(small):
    with pytest.raises(ValueError):
        solve(replace(small, lead_times=(2, 4), lead_probs=(0.5, 0.5)), first_wday=1)
