import json
import os
import subprocess
import sys

import pandas as pd


def run(*args, cwd):
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=cwd, env=env)


def test_end_to_end_on_synthetic_lines(csv_folder, tmp_path):
    data, artifacts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    run("customer_segments.data", "--source", str(csv_folder), "--data-dir", str(data), cwd=tmp_path)
    common = ["--data-dir", str(data), "--artifacts", str(artifacts), "--quick"]
    run("customer_segments.train", *common, cwd=tmp_path)
    stamp = (artifacts / "stability.csv").stat().st_mtime_ns
    # A second run finds every stage stored and fits nothing again.
    run("customer_segments.train", *common, cwd=tmp_path)
    assert (artifacts / "stability.csv").stat().st_mtime_ns == stamp
    run("customer_segments.evaluate", "--artifacts", str(artifacts), "--quick", cwd=tmp_path)
    run("customer_segments.report", "--data-dir", str(data), "--artifacts", str(artifacts),
        "--reports", str(reports), "--data-label", "synthetic", cwd=tmp_path)  # fmt: skip

    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    assert 3 <= selection["k"] <= 5
    comparison = pd.read_csv(artifacts / "evaluation" / "comparison.csv").set_index("segmentation")
    assert set(comparison.index) == {
        "kmeans_rfm_mix",
        "kmeans_first_attempt",
        "kmeans_rfm",
        "hdbscan",
        "rfm_grid",
    }
    # The synthetic types differ in how often they buy, so every segmentation explains some of it.
    assert (comparison["eta2_purchase"] > 0).all()
    profiles = pd.read_csv(artifacts / "evaluation" / "profiles.csv")
    assert profiles["customers"].sum() == selection["customers"]
    assert profiles["spend_share"].is_monotonic_decreasing
    ablation = pd.read_csv(artifacts / "evaluation" / "profiles_first_attempt.csv")
    assert ablation["customers"].sum() == selection["customers"]
    assert len(list((reports / "figures").glob("*.png"))) == 5
    html = (reports / "segment_map.html").read_text()
    assert "Segment 1" in html and "customer_id" not in html
    assert "## Next quarter check" in (reports / "results.md").read_text()
