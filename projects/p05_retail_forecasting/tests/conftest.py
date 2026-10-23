import pytest

from retail_forecast.synthetic import make_tables, write_archive


@pytest.fixture(scope="session")
def raw_tables():
    return make_tables(n_days=400, items_per_dept=4)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Processed tables of a synthetic archive (54 series, 500 days)."""
    from retail_forecast.data import normalize, read_archive, write
    from retail_forecast.demand import write_demand_tables

    root = tmp_path_factory.mktemp("m5")
    archive = write_archive(root / "m5_synthetic.zip")
    write(normalize(read_archive(archive), expected_series=None), root / "data")
    write_demand_tables(root / "data")
    return root / "data"


@pytest.fixture(scope="session")
def tables(data_dir):
    from retail_forecast.data import load

    return load(data_dir)
