import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from flow_anomaly import autoencoder, detectors
from flow_anomaly.config import QUICK_AUTOENCODER, QUICK_FOREST, QUICK_SUPERVISED, TTL_FEATURES
from flow_anomaly.features import FlowPreprocessor
from flow_anomaly.train import fit_sample


@pytest.fixture(scope="module")
def matrices(flows):
    sample = fit_sample(flows, 0.01, 2_000, seed=0)
    prep = FlowPreprocessor().fit(flows.iloc[sample])
    test = flows.loc[flows["split"] == "test"]
    return prep, prep.transform(flows.iloc[sample]), prep.transform(test), test["label"].to_numpy()


def test_fit_sample_has_the_attack_share(flows):
    sample = fit_sample(flows, 0.05, 1_000, seed=0)
    assert len(sample) == 1_000
    assert flows["label"].to_numpy()[sample].mean() == 0.05
    assert (flows["split"].to_numpy()[sample] == "train").all()
    assert len(np.unique(sample)) == len(sample)


def test_preprocessor_maps_unseen_values_to_other(matrices, flows):
    prep = matrices[0]
    row = flows.iloc[:1].copy()
    row["proto"] = "never_seen"
    x = prep.transform(row)
    names = prep.feature_names
    assert x.shape == (1, len(names))
    assert x[0, names.index("proto=other")] == 1.0


def test_preprocessor_without_ttl_features(flows):
    numeric = [c for c in FlowPreprocessor().numeric if c not in TTL_FEATURES]
    prep = FlowPreprocessor(numeric=numeric).fit(flows)
    assert not set(TTL_FEATURES) & set(prep.feature_names)


@pytest.mark.parametrize("name", ["robust_z", "iforest", "lof", "hbos", "ecod", "copod", "autoencoder"])
def test_detectors_rank_attacks_above_normal_flows(name, matrices, tmp_path):
    _, x_fit, x_test, y_test = matrices
    model = detectors.make(name, QUICK_FOREST, QUICK_AUTOENCODER, tmp_path).fit(x_fit)
    scores = model.score(x_test)
    assert scores.shape == (len(x_test),)
    assert roc_auc_score(y_test, scores) > 0.6


def test_ensemble_and_supervised(matrices, flows, tmp_path):
    prep, x_fit, x_test, y_test = matrices
    members = {
        name: detectors.make(name, QUICK_FOREST, QUICK_AUTOENCODER, tmp_path).fit(x_fit)
        for name in detectors.ENSEMBLE_MEMBERS
    }
    ensemble = detectors.make("ensemble", QUICK_FOREST, QUICK_AUTOENCODER, tmp_path, members).fit(x_fit)
    scores = ensemble.score(x_test)
    assert ((scores >= 0) & (scores <= 1)).all()
    train = flows.loc[flows["split"] == "train"]
    val = flows.loc[flows["split"] == "validation"]
    supervised = detectors.Supervised(QUICK_SUPERVISED).fit(
        prep.transform(train), train["label"].to_numpy(), prep.transform(val), val["label"].to_numpy()
    )
    assert roc_auc_score(y_test, supervised.score(x_test)) > 0.9


def test_autoencoder_resumes_after_the_last_saved_epoch(matrices, tmp_path):
    x = matrices[1]
    settings = QUICK_AUTOENCODER.__class__(**{**QUICK_AUTOENCODER.__dict__, "epochs": 2, "patience": 10})
    first = autoencoder.fit(x, settings, tmp_path)
    assert first["epoch"] == 2
    longer = settings.__class__(**{**settings.__dict__, "epochs": 3})
    second = autoencoder.fit(x, longer, tmp_path)
    assert second["epoch"] == 3
    assert [h["epoch"] for h in second["history"]] == [1, 2, 3]
    assert second["history"][:2] == first["history"]
    pd.read_json(tmp_path / "history.json")  # written and readable
