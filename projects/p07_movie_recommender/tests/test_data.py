import pandas as pd
import pytest

from movie_recs.data import SchemaError, check, movie_table, normalize, read_archive, time_split
from movie_recs.interactions import make_stage


def test_archive_reads_and_normalizes(archive):
    tables = read_archive(archive)
    processed = normalize(tables, expect_counts=False)
    ratings = processed["ratings"]
    assert list(ratings.columns) == ["user", "item", "rating", "ts"]
    assert ratings["ts"].is_monotonic_increasing
    assert len(ratings) == len(tables["ratings"])
    split = processed["split"]
    assert split["validation_start"] < split["test_start"] <= split["end"]


def test_release_counts_are_checked(archive):
    with pytest.raises(SchemaError, match="expected"):
        check(read_archive(archive), expect_counts=True)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t: t["ratings"].assign(rating=t["ratings"]["rating"] + 0.1), "half stars"),
        (lambda t: pd.concat([t["ratings"], t["ratings"].head(1)]), "twice"),
        (lambda t: t["ratings"].assign(movieId=t["ratings"]["movieId"] + 10_000), "lacks"),
    ],
)
def test_bad_ratings_are_refused(archive, change, message):
    tables = read_archive(archive)
    tables["ratings"] = change(tables)
    with pytest.raises(SchemaError, match=message):
        check(tables, expect_counts=False)


def test_movie_table_parses_year_and_genres():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Untitled"],
            "genres": ["Adventure|Animation|Children", "(no genres listed)"],
        }
    )
    table = movie_table(movies)
    assert table["year"].iloc[0] == 1995 and pd.isna(table["year"].iloc[1])
    assert table.loc[0, ["Adventure", "Animation", "Children", "Comedy"]].tolist() == [1, 1, 1, 0]
    assert table.loc[1, "Drama"] == 0


def test_time_split_quantiles():
    split = time_split(pd.Series(range(1000)).to_numpy())
    assert (split["validation_start"], split["test_start"], split["end"]) == (959, 979, 999)


def test_stages_never_see_their_window(data):
    for name in ("validation", "test"):
        stage = make_stage(data["ratings"], data["split"], name)
        assert stage.history["ts"].max() < stage.start
        assert stage.window["ts"].min() >= stage.start and stage.window["ts"].max() <= stage.end
        assert stage.targets.merge(stage.history, on=["user", "item"]).empty
    validation = make_stage(data["ratings"], data["split"], "validation")
    test = make_stage(data["ratings"], data["split"], "test")
    assert validation.end < test.start
    assert len(validation.history) + len(validation.window) == len(test.history)
