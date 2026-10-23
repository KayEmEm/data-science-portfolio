import numpy as np
import pandas as pd
import pytest
from scipy import stats

from email_uplift.experiment import (
    balance_table,
    cuped_adjust,
    difference_in_means,
    effects_table,
    minimum_detectable_effect,
    required_sample_size,
    sample_ratio_check,
    variance_reduction_table,
)
from email_uplift.synthetic import VISIT_LIFT


def test_difference_in_means_matches_welch():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 2.0, 4000), rng.normal(0.5, 1.0, 5000)
    est = difference_in_means(a, b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert est["effect"] == pytest.approx(a.mean() - b.mean())
    assert est["se"] == pytest.approx(se)
    assert est["ci_high"] - est["ci_low"] == pytest.approx(2 * stats.norm.ppf(0.975) * se)
    assert est["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)


def test_sample_ratio_check_flags_an_unequal_split(customers):
    assert sample_ratio_check(customers)["p_value"] > 0.001
    skewed = pd.concat([customers, customers[customers["segment"] == "Mens E-Mail"].head(600)])
    assert sample_ratio_check(skewed)["p_value"] < 0.001


def test_randomized_arms_are_balanced(customers):
    table = balance_table(customers)
    assert table.filter(like="smd_").abs().max().max() < 0.1


def test_effects_recover_the_visit_lift(customers):
    table = effects_table(customers).set_index(["arm", "outcome"])
    for arm in ("mens", "womens"):
        share = customers[arm].mean()
        expected = share * VISIT_LIFT[arm] + (1 - share) * 0.01
        row = table.loc[(arm, "visit")]
        assert row["ci_low"] < expected < row["ci_high"]
    assert (table["p_holm"] >= table["p_value"]).all()


def test_mde_matches_the_normal_formula():
    sd, n = 15.0, 20000
    expected = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd * np.sqrt(2 / n)
    assert minimum_detectable_effect(sd, n, n) == pytest.approx(expected, rel=1e-3)
    assert required_sample_size(expected, sd) == pytest.approx(n, rel=1e-3)


def test_cuped_keeps_the_mean_and_cuts_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 20, 10000)
    y = 0.5 * x + rng.normal(0, 5, 10000)
    adjusted, theta = cuped_adjust(y, x)
    assert theta == pytest.approx(0.5, abs=0.02)
    assert adjusted.mean() == pytest.approx(y.mean())
    # the residual variance is 5 squared; the raw variance adds 0.5 squared times 20 squared
    assert adjusted.var() == pytest.approx(25, rel=0.05)
    assert y.var() == pytest.approx(125, rel=0.05)


def test_variance_reduction_keeps_the_estimates(customers):
    table = variance_reduction_table(customers).set_index(["arm", "outcome"])
    assert (table.loc[(slice(None), "spend"), "variance_reduction_cuped"] > 0).all()
    assert np.allclose(table["effect_raw"], table["effect_cuped"], atol=3 * table["se_raw"])
