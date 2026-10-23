import json

import joblib
from fastapi.testclient import TestClient

from loan_default.evaluate import run as evaluate


def test_training_outputs(trained):
    artifacts, summary = trained
    assert summary["data"] == "synthetic"
    assert set(summary["models"]) == {"logistic_regression", "lightgbm", "xgboost"}
    for result in summary["models"].values():
        assert 0.5 < result["test_calibrated"]["roc_auc"] <= 1.0
        assert 0.0 <= result["threshold"] <= 1.0
    for name in ["model.joblib", "metrics.json", "test_predictions.parquet", "example_request.json"]:
        assert (artifacts / name).exists()


def test_calibrated_predictions_are_probabilities(trained, features):
    artifacts, _ = trained
    model = joblib.load(artifacts / "model.joblib")["model"]
    p = model.predict(features.head(50))
    assert ((p >= 0) & (p <= 1)).all()


def test_evaluate_writes_reports(trained, tmp_path):
    artifacts, _ = trained
    evaluate(artifacts, tmp_path)
    for name in ["model_comparison.md", "fairness_age.md", "results.json"]:
        assert (tmp_path / name).exists()
    for fig in ["threshold_analysis.png", "calibration.png", "explanations.png"]:
        assert (tmp_path / "figures" / fig).stat().st_size > 0


def test_api_scores_example(trained, monkeypatch):
    artifacts, _ = trained
    monkeypatch.setenv("MODEL_PATH", str(artifacts / "model.joblib"))
    from loan_default import api

    api.load_bundle.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    body = json.loads((artifacts / "example_request.json").read_text())
    response = client.post("/score", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["probability_of_default"] <= 1.0
    assert payload["decision"] in {"approve", "reject"}
    assert payload["unknown_features"] == []

    sparse = client.post("/score", json={"features": {"AMT_CREDIT": 500000.0, "NOT_A_COLUMN": 1}}).json()
    assert sparse["unknown_features"] == ["NOT_A_COLUMN"]
