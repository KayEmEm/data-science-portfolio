import pytest

from loan_default.config import CostAssumptions, RunSettings
from loan_default.features import build_features
from loan_default.synthetic import write_tables


@pytest.fixture(scope="session")
def parquet_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("parquet")
    write_tables(path, n_applicants=1500)
    return path


@pytest.fixture(scope="session")
def features(parquet_dir):
    return build_features(parquet_dir)


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
