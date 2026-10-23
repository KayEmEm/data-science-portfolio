import numpy as np
import pytest

from retail_forecast.hierarchy import (
    bottom_shares,
    build_hierarchy,
    disaggregate,
    store_dept_frame,
    store_dept_summing_matrix,
)


@pytest.fixture(scope="module")
def levels(tables):
    return build_hierarchy(tables["series"])


def test_level_sizes(levels):
    # 3 stores in 2 states, 3 departments in 2 categories, 6 items per department
    sizes = {n: lv.matrix.shape[0] for n, lv in levels.items()}
    assert sizes == {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 4, 7: 6, 8: 6, 9: 9, 10: 18, 11: 36, 12: 54}


def test_every_level_sums_to_the_total(levels, tables):
    total = levels[1].aggregate(tables["sales"])
    for level in levels.values():
        assert np.allclose(level.aggregate(tables["sales"]).sum(axis=0), total[0])


def test_summing_matrix_matches_direct_aggregation(levels, tables):
    summing, owners, names = store_dept_summing_matrix(tables["series"])
    store_dept = levels[9].aggregate(tables["sales"])
    direct = np.vstack([levels[n].aggregate(tables["sales"]) for n in range(1, 10)])
    assert np.allclose(summing @ store_dept, direct)
    assert owners.count(9) == 9 and len(names) == summing.shape[0]
    assert names[-9:] == levels[9].names
    assert len(store_dept_frame(tables["series"])) == 9


def test_top_down_split_keeps_the_aggregate(levels, tables):
    level = levels[9]
    shares = bottom_shares(tables["sales"][:, :300], level)
    assert np.allclose(np.asarray(level.matrix @ shares).ravel(), 1.0)
    forecast = np.arange(9 * 28, dtype=float).reshape(9, 28)
    assert np.allclose(level.aggregate(disaggregate(forecast, level, shares)), forecast)


def test_split_is_even_for_an_aggregate_without_recent_sales(levels):
    level = levels[9]
    shares = bottom_shares(np.zeros((54, 40)), level)
    assert np.allclose(shares, 1 / 6)
