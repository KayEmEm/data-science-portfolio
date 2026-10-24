import pytest

from road_damage.synthetic import write_archives


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    """Synthetic RDD2022 zips for Czech and India, 40 labelled images each."""
    return write_archives(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def data_dir(source, tmp_path_factory):
    from road_damage.data import build

    root = tmp_path_factory.mktemp("data")
    build(source, root, block=4)
    return root
