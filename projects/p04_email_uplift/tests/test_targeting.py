import numpy as np
import pytest

from email_uplift.config import Economics
from email_uplift.targeting import (
    bootstrap_policy_values,
    choose_arms,
    policy_value,
    qini_coefficient,
    qini_curve,
    rewards,
)


@pytest.fixture
def experiment():
    """Half treated; only customers with high ``x`` respond to treatment."""
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.random(n)
    treated = rng.random(n) < 0.5
    y = rng.binomial(1, 0.1 + 0.2 * treated * (x > 0.7)).astype(float)
    return x, y, treated


def test_qini_curve_starts_at_zero_and_ends_at_the_average_effect(experiment):
    x, y, treated = experiment
    shares, q = qini_curve(x, y, treated)
    assert shares[0] == 0 and shares[-1] == 1 and q[0] == 0
    effect = y[treated].mean() - y[~treated].mean()
    assert q[-1] == pytest.approx(effect, abs=1e-9)


def test_qini_coefficient_rewards_the_right_ranking(experiment):
    x, y, treated = experiment
    # The oracle ranking reaches the full effect (0.06) after the top 30 percent: area about 0.02
    assert qini_coefficient(x, y, treated) > 0.01
    assert qini_coefficient(-x, y, treated) < -0.01
    random_scores = np.random.default_rng(0).random(len(x))
    assert abs(qini_coefficient(random_scores, y, treated)) < 0.01


def test_choose_arms_sends_only_when_margin_covers_cost():
    effects = np.array([[1.0, 0.2], [0.1, 0.2], [0.0, 2.0], [0.3, 0.3]])
    arms = choose_arms(effects, Economics(margin=0.3, cost_per_email=0.1))
    assert arms.tolist() == [1, 0, 2, 0]


def test_policy_value_of_a_uniform_policy_is_that_arm_mean():
    rng = np.random.default_rng(4)
    t = rng.integers(0, 3, 3000)
    spend = rng.exponential(5, 3000) * (t + 1)
    reward = rewards(spend, t, Economics())
    for arm in range(3):
        assert policy_value(np.full(3000, arm), t, reward) == pytest.approx(reward[t == arm].mean())


def test_rewards_charge_the_email_cost():
    r = rewards(np.array([10.0, 10.0]), np.array([0, 2]), Economics(margin=0.3, cost_per_email=0.1))
    assert r.tolist() == pytest.approx([3.0, 2.9])


def test_bootstrap_draws_are_paired_and_reproducible():
    rng = np.random.default_rng(5)
    t = rng.integers(0, 3, 900)
    reward = rng.normal(size=900)
    policies = {"a": np.zeros(900, int), "b": np.zeros(900, int)}
    first = bootstrap_policy_values(policies, t, reward, 20, seed=1)
    again = bootstrap_policy_values(policies, t, reward, 20, seed=1)
    assert np.array_equal(first["a"], first["b"])
    assert np.array_equal(first["a"], again["a"])
