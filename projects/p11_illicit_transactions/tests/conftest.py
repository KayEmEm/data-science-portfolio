import pytest

from illicit_graph.synthetic import write_release


@pytest.fixture(scope="session")
def release_folder(tmp_path_factory):
    return write_release(tmp_path_factory.mktemp("release"), 6_000)


@pytest.fixture(scope="session")
def prepared(release_folder):
    from illicit_graph.data import prepare

    return prepare(release_folder)
