import pytest

from email_uplift.synthetic import write_customers


@pytest.fixture(scope="session")
def customers_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "customers.parquet"
    write_customers(path, n=6000)
    return path


@pytest.fixture(scope="session")
def customers(customers_path):
    from email_uplift.data import load

    return load(customers_path)
