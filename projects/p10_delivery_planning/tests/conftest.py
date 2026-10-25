import pytest

from delivery_planning.data import load
from delivery_planning.synthetic import write


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("source")
    write(folder, customers=20)
    return folder


@pytest.fixture(scope="session")
def r101(source):
    return load(source, "R101")
