import json
import os
import subprocess
import sys

import pandas as pd


def run(*args, cwd):
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=cwd, env=env)


def test_end_to_end_on_synthetic_flows(full_folder, tmp_path):
    data, artifacts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    run("flow_anomaly.data", "--source", str(full_folder), "--data-dir", str(data), cwd=tmp_path)
    common = ["--data-dir", str(data), "--artifacts", str(artifacts)]
    run("flow_anomaly.train", *common, "--quick", cwd=tmp_path)
    stored = sorted((artifacts / "main" / "scores").iterdir())
    # A second run finds every detector stored and fits nothing again.
    run("flow_anomaly.train", *common, "--quick", cwd=tmp_path)
    assert sorted((artifacts / "main" / "scores").iterdir()) == stored
    run("flow_anomaly.evaluate", *common, "--data-label", "synthetic", "--bootstrap", "5", cwd=tmp_path)
    run("flow_anomaly.report", *common, "--reports", str(reports), cwd=tmp_path)

    summary = pd.read_csv(artifacts / "evaluation" / "summary.csv")
    assert {"main", "no_ttl", "contamination_0", "contamination_0.2"} <= set(summary["experiment"])
    main = summary.loc[summary["experiment"] == "main"].set_index("detector")
    assert main["test_fpr"].max() < 0.05
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    assert (main["test_pr_auc"] > selection["test_attack_share"]).all()
    assert selection["detector"] != "lightgbm"
    assert (reports / "triage_example.md").read_text().count("\n| ") >= 20
    assert len(list((reports / "figures").glob("*.png"))) == 5
