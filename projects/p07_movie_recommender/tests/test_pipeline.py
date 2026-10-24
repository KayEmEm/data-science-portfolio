import json

import numpy as np
import pandas as pd

from movie_recs import evaluate, rank, report, train
from movie_recs.demo import load_lists_by_model, user_view
from movie_recs.interactions import make_stage


def run(module, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    module.main()


def test_end_to_end_on_synthetic_data(data_dir, data, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    artifacts = tmp_path / "artifacts"
    common = ["--data-dir", str(data_dir), "--artifacts", str(artifacts)]
    run(train, [*common, "--quick"], monkeypatch)
    run(rank, [*common, "--quick"], monkeypatch)
    run(evaluate, [*common, "--data-label", "synthetic"], monkeypatch)
    run(report, [*common, "--reports", str(tmp_path / "reports")], monkeypatch)

    summary = pd.read_csv(artifacts / "evaluation" / "summary.csv")
    assert set(summary["model"]) == set(evaluate.MODELS)
    assert set(summary["stage"]) == {"validation", "test"}
    test = summary.loc[summary["stage"] == "test"].set_index("model")
    assert test["recall@10"].between(0, 1).all()
    # The synthetic users choose by taste; the learned models must beat the all time popularity list.
    assert test.loc["als", "ndcg@10"] > test.loc["popularity", "ndcg@10"]
    assert test.loc["ranker", "ndcg@10"] > test.loc["popularity", "ndcg@10"]

    selection = json.loads((artifacts / "selection.json").read_text())
    assert {"als", "two_tower"} <= set(selection)
    results = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert results["data"] == "synthetic"
    figures = {p.name for p in (tmp_path / "reports" / "figures").glob("*.png")}
    assert figures == {
        "metrics_by_model.png",
        "coverage_novelty.png",
        "cold_start_users.png",
        "two_tower_epochs.png",
        "ranker_importance.png",
    }

    # A second run skips every stored step and leaves the lists unchanged.
    before = np.load(artifacts / "test" / "lists" / "als.npz")["items"]
    run(train, [*common, "--quick"], monkeypatch)
    assert np.array_equal(before, np.load(artifacts / "test" / "lists" / "als.npz")["items"])

    stage = make_stage(data["ratings"], data["split"], "test")
    lists = load_lists_by_model(artifacts)
    view = user_view(stage, data["movies"], lists, int(lists["ranker"][0][0]))
    assert list(view["recommendations"].columns) == evaluate.MODELS
    assert len(view["recommendations"]) == 10
