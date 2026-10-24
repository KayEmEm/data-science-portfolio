import numpy as np
import pytest
import torch

from movie_recs import als, baselines, two_tower
from movie_recs.config import QUICK_TWO_TOWER, ALSSettings
from movie_recs.metrics import PAD, per_user, top_n


def seen_by(stage, user):
    return set(stage.history.loc[stage.history["user"] == user, "item"])


def test_popularity_and_als_exclude_seen_movies(stage):
    users = stage.eval_users()[:20]
    pop_items, _ = baselines.recommend(stage, users, 10, days=None)
    model = als.fit(stage, ALSSettings(factors=8, iterations=5), seed=0)
    als_items, _ = als.recommend(model, stage, users, 10)
    for items in (pop_items, als_items):
        for user, row in zip(users, items, strict=True):
            assert not (set(row) & seen_by(stage, user))
            assert (row != PAD).all()


def test_als_beats_popularity_on_taste_driven_data(stage):
    users = stage.eval_users()
    pop_items, _ = baselines.recommend(stage, users, 10, days=None)
    model = als.fit(stage, ALSSettings(factors=16, iterations=10), seed=0)
    als_items, _ = als.recommend(model, stage, users, 10)
    pop = per_user(users, pop_items, stage.targets, 10)["recall"].mean()
    learned = per_user(users, als_items, stage.targets, 10)["recall"].mean()
    assert learned > pop


def test_training_histories_hold_only_earlier_movies(stage):
    sequences = two_tower.Sequences(stage)
    positions = sequences.sample(5, np.random.default_rng(0))
    starts = sequences.start[sequences.row_of[positions]]
    histories = sequences.histories(positions, starts, 10)
    for position, history in zip(positions[:200], histories[:200], strict=True):
        start = sequences.start[sequences.row_of[position]]
        expected = sequences.items[max(start, position - 10) : position]
        assert history[history > 0].tolist() == expected.tolist()
        assert sequences.liked[position]


def test_sample_respects_the_limit_per_user(stage):
    sequences = two_tower.Sequences(stage)
    positions = sequences.sample(3, np.random.default_rng(1))
    assert np.bincount(sequences.row_of[positions]).max() <= 3
    assert len(np.unique(positions)) == len(positions)


@pytest.fixture(scope="module")
def trained(stage, data, tmp_path_factory):
    new = np.setdiff1d(stage.targets["item"].unique(), stage.items)
    features = two_tower.ItemFeatures(stage, data["movies"], data["tags"], QUICK_TWO_TOWER.n_tags, extra=new)
    directory = tmp_path_factory.mktemp("tt")
    model = two_tower.train(stage, features, QUICK_TWO_TOWER, seed=0, epochs=2, directory=directory)
    return model, features, directory


def test_two_tower_faiss_matches_exact_scoring(stage, trained):
    model, _, _ = trained
    users = stage.eval_users()[:30]
    found, _ = two_tower.recommend(model, stage, users, 10, QUICK_TWO_TOWER, batch=7)
    user_vectors, item_vectors = two_tower.embed(model, stage, users, QUICK_TWO_TOWER)
    exact, _ = top_n(
        lambda rows: user_vectors[np.searchsorted(stage.user_rows(users), rows)] @ item_vectors.T,
        stage.user_rows(users),
        stage.matrix,
        stage.items,
        10,
    )
    assert (found == exact).mean() > 0.95  # ties may order differently
    for user, row in zip(users, found, strict=True):
        assert not (set(row) & seen_by(stage, user))


def test_two_tower_resumes_from_its_last_checkpoint(stage, trained, capsys):
    model, features, directory = trained
    again = two_tower.train(stage, features, QUICK_TWO_TOWER, seed=0, epochs=2, directory=directory)
    assert "resumed after epoch 2" in capsys.readouterr().out
    for a, b in zip(model.state_dict().values(), again.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert [r["epoch"] for r in two_tower.epoch_records(directory)] == [1, 2]


def test_new_movies_get_the_unknown_id_and_their_features(trained):
    _, features, _ = trained
    n = len(features.catalog)
    assert (features.ids[n + 1 :] == 0).all()
    assert (features.ids[1 : n + 1] == np.arange(1, n + 1)).all()
    assert features.genres[n + 1 :].sum() > 0 or len(features.extra) == 0
