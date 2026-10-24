import pytest

from flow_anomaly.synthetic import make_flows, write_full, write_partition


@pytest.fixture(scope="session")
def raw_flows():
    return make_flows(6_000)


@pytest.fixture(scope="session")
def full_folder(raw_flows, tmp_path_factory):
    return write_full(tmp_path_factory.mktemp("full"), raw_flows)


@pytest.fixture(scope="session")
def partition_folder(raw_flows, tmp_path_factory):
    return write_partition(tmp_path_factory.mktemp("partition"), raw_flows)


@pytest.fixture(scope="session")
def flows(full_folder):
    from flow_anomaly.data import prepare

    return prepare(full_folder)[0]
