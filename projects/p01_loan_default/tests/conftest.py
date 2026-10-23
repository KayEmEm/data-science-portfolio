import pytest

from loan_default.config import CostAssumptions, RunSettings
from loan_default.features import build_features
from loan_default.synthetic import write_clients


@pytest.fixture(scope="session")
def clients_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "clients.parquet"
    write_clients(path, n_clients=2000)
    return path


@pytest.fixture(scope="session")
def features(clients_path):
    return build_features(clients_path)


@pytest.fixture(scope="session")
def trained(features, tmp_path_factory):
    from loan_default.train import run

    artifacts = tmp_path_factory.mktemp("artifacts")
    features_path = artifacts / "features.parquet"
    features.to_parquet(features_path, index=False)
    summary = run(
        features_path,
        artifacts,
        RunSettings.quick(),
        CostAssumptions(),
        "synthetic",
        ["logistic_regression", "lightgbm", "xgboost"],
    )
    return artifacts, summary
