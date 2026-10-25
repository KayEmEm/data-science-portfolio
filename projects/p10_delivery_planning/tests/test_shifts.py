import pandas as pd
import pytest

from delivery_planning import shifts as sh
from delivery_planning.config import CHECK_MINUTES, DAY_START, MIN_PAID_HOURS, OVERTIME_AFTER


def duty_frame(rows):
    table = pd.DataFrame(rows, columns=["duty", "day", "start", "end"])
    table["hours"] = (table["end"] - table["start"]) / 60
    table["customers"] = 1
    return table


def one_driver(**changes):
    row = {
        "driver": "D01", "contract": "full_time", "wage": 30.0, "guarantee": 0.0, "max_hours": 48.0,
        "max_days": 5, "window": "day", "earliest": 360.0, "latest": 1260.0, "days": "1111111",
    }  # fmt: skip
    row.update(changes)
    return pd.DataFrame([row])


def test_duties_map_the_depot_horizon_onto_the_day(r101):
    instance, best = r101
    table = sh.duties(0, instance, best["routes"])
    assert len(table) == len(best["routes"])
    assert (table["start"] >= DAY_START - CHECK_MINUTES).all()
    assert (table["end"] <= DAY_START + 600 + CHECK_MINUTES + 1e-9).all()
    assert (table["hours"] > 2 * CHECK_MINUTES / 60).all()


def test_driver_pool_is_reproducible():
    first, second = sh.make_drivers(20), sh.make_drivers(20)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["contract"]) == {"full_time", "part_time"}
    assert first["days"].str.len().eq(7).all()


def test_pay_has_minimum_shift_guarantee_and_overtime():
    duties = duty_frame([("a", 0, 480, 540), ("b", 0, 600, 660), ("c", 1, 480, 1080)])
    priced = sh.roster_cost({"a": "D01", "b": "D01"}, duties, one_driver())
    # One shift from 08:00 to 11:00 is paid for 4 hours; duty c goes to the agency at 10 hours.
    assert priced["paid_hours"] == MIN_PAID_HOURS
    assert priced["driver_pay"] == pytest.approx(30 * MIN_PAID_HOURS)
    assert priced["agency"] == pytest.approx(sh.agency_cost(10.0))
    guaranteed = sh.roster_cost({}, duties, one_driver(guarantee=30.0))
    assert guaranteed["driver_pay"] == pytest.approx(30 * 30)
    long_week = duty_frame([(f"d{d}", d, 420, 1080) for d in range(5)])
    over = sh.roster_cost({f"d{d}": "D01" for d in range(5)}, long_week, one_driver())
    hours = 5 * 11.0
    assert over["driver_pay"] == pytest.approx(30 * (hours + 0.5 * (hours - OVERTIME_AFTER)))


def test_check_roster_finds_overlap_and_window_violations():
    duties = duty_frame([("a", 0, 480, 600), ("b", 0, 570, 700), ("c", 0, 300, 400)])
    drivers = one_driver()
    assert any("overlapping" in p for p in sh.check_roster({"a": "D01", "b": "D01"}, duties, drivers))
    assert any("cannot take" in p for p in sh.check_roster({"c": "D01"}, duties, drivers))
    assert sh.check_roster({"a": "D01"}, duties, drivers) == []


def test_mip_with_cbc_beats_or_matches_greedy_and_keeps_every_rule(r101):
    instance, best = r101
    duties = pd.concat([sh.duties(d, instance, best["routes"]) for d in range(2)], ignore_index=True)
    drivers = sh.make_drivers(8)
    greedy = sh.greedy(duties, drivers)
    assert sh.check_roster(greedy, duties, drivers) == []
    model = sh.build_model(duties, drivers)
    record = sh.solve_model(model, "cbc", 60)
    assert record["status"] == "optimal"
    assert sh.check_roster(record["assignment"], duties, drivers) == []
    priced = sh.roster_cost(record["assignment"], duties, drivers)["total"]
    assert priced == pytest.approx(record["objective"], rel=1e-6)
    assert priced <= sh.roster_cost(greedy, duties, drivers)["total"] + 1e-6


def test_highs_refuses_to_run_next_to_ortools():
    import ortools  # noqa: F401  (loaded by the routing tests in this process anyway)

    with pytest.raises(RuntimeError, match="OR-Tools"):
        sh.solver("highs", 1)
