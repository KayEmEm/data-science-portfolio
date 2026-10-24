import pytest

from customer_segments.synthetic import make_lines, write_csv


@pytest.fixture(scope="session")
def raw_lines():
    return make_lines(300)


@pytest.fixture(scope="session")
def csv_folder(raw_lines, tmp_path_factory):
    folder = tmp_path_factory.mktemp("csv")
    write_csv(folder, raw_lines)
    return folder


@pytest.fixture(scope="session")
def lines(csv_folder):
    from customer_segments.data import prepare

    return prepare(csv_folder)[0]
