import json
import os
import subprocess
import sys

import pandas as pd


def run(*args, cwd):
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONWARNINGS": "ignore"}
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=cwd, env=env)


def test_end_to_end_on_a_synthetic_graph(release_folder, tmp_path):
    data, artifacts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    run("illicit_graph.data", "--source", str(release_folder), "--data-dir", str(data), cwd=tmp_path)
    common = ["--data-dir", str(data), "--artifacts", str(artifacts)]
    run("illicit_graph.network", *common, "--samples", "8", cwd=tmp_path)
    run("illicit_graph.train", *common, "--quick", cwd=tmp_path)
    stored = {p: p.stat().st_mtime_ns for p in (artifacts / "models").rglob("scores.npy")}
    # A second run finds every model stored and fits nothing again.
    run("illicit_graph.train", *common, "--quick", cwd=tmp_path)
    assert {p: p.stat().st_mtime_ns for p in (artifacts / "models").rglob("scores.npy")} == stored
    assert len(stored) == 7
    run("illicit_graph.evaluate", *common, "--data-label", "synthetic", "--bootstrap", "5", cwd=tmp_path)
    run("illicit_graph.report", *common, "--reports", str(reports), cwd=tmp_path)
    run("illicit_graph.neo4j_export", "export", *common, cwd=tmp_path)

    summary = pd.read_csv(artifacts / "evaluation" / "summary.csv").set_index("model")
    assert summary["test_f1"].between(0, 1).all()
    assert summary.loc["lightgbm_all", "test_f1"] > summary.loc["logistic_local", "test_f1"]
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    assert selection["model"] in summary.index
    assert len(list((reports / "figures").glob("*.png"))) == 6
    assert "## Models" in (reports / "results.md").read_text()
    exported = pd.read_csv(artifacts / "neo4j" / "transactions.csv")
    assert len(exported) == 6_000
    assert set(exported["class"]) <= {"illicit", "licit", "unknown"}
