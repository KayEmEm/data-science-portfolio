import pytest

from inventory_rl.dynamics import Problem
from inventory_rl.synthetic import write_archive


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    return write_archive(tmp_path_factory.mktemp("raw") / "m5_synthetic.zip")


@pytest.fixture(scope="session")
def prepared(archive, tmp_path_factory):
    from inventory_rl.data import write

    data_dir = tmp_path_factory.mktemp("data")
    write(archive, data_dir)
    return data_dir


@pytest.fixture
def problem():
    return Problem(
        weekday_mean=(6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 5.0),
        dispersion=5.0,
        holding=0.01,
        lost_sale=1.5,
        order_cost=10.0,
        capacity=150,
        horizon=60,
        name="test",
    )


def constant_problem(units: int, lead: int, **changes) -> Problem:
    """Demand of exactly ``units`` every day (a very large dispersion is near Poisson, so the
    scenarios below are built by hand) and one fixed lead time."""
    values = {
        "weekday_mean": (float(units),) * 7,
        "dispersion": 1e4,
        "holding": 0.01,
        "lost_sale": 2.0,
        "order_cost": 5.0,
        "capacity": 200,
        "lead_times": (lead,),
        "lead_probs": (1.0,),
        "horizon": 70,
        "name": "constant",
    }
    return Problem(**(values | changes))
