import pytest

from movie_recs.synthetic import write_archive


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    return write_archive(
        tmp_path_factory.mktemp("ml") / "ml-32m.zip", n_users=600, n_movies=300, mean_ratings=40
    )


@pytest.fixture(scope="session")
def data_dir(archive, tmp_path_factory):
    from movie_recs.data import normalize, read_archive, write

    root = tmp_path_factory.mktemp("data")
    write(normalize(read_archive(archive), expect_counts=False), root)
    return root


@pytest.fixture(scope="session")
def data(data_dir):
    from movie_recs.data import load

    return load(data_dir)


@pytest.fixture(scope="session")
def stage(data):
    from movie_recs.interactions import make_stage

    return make_stage(data["ratings"], data["split"], "validation")
