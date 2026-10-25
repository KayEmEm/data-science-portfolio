import json
import os
import subprocess
import sys

import pandas as pd


def run(*args, cwd):
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONWARNINGS": "ignore"}
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=cwd, env=env)


def test_end_to_end_on_a_synthetic_archive(archive, tmp_path):
    data, artifacts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    run("inventory_rl.data", "--source", str(archive), "--data-dir", str(data), cwd=tmp_path)
    common = ["--data-dir", str(data), "--artifacts", str(artifacts)]
    run("inventory_rl.baselines", *common, "--quick", cwd=tmp_path)
    run("inventory_rl.dp", *common, "--quick", cwd=tmp_path)
    run("inventory_rl.train", *common, "--quick", "--jobs", "2", cwd=tmp_path)
    stored = {p: p.stat().st_mtime_ns for p in (artifacts / "runs").rglob("info.json")}
    assert (
        len(stored) == 9
    )  # 3 products x 2 algorithms x 1 seed, 2 shaping runs, 1 run with a fixed lead time
    # A second run finds every run stored and trains nothing again.
    run("inventory_rl.train", *common, "--quick", cwd=tmp_path)
    assert {p: p.stat().st_mtime_ns for p in (artifacts / "runs").rglob("info.json")} == stored
    run("inventory_rl.dp", *common, "--quick", cwd=tmp_path)  # again, now with the fixed lead time PPO run
    assert "ppo" in json.loads((artifacts / "dp" / "result.json").read_text())
    run("inventory_rl.evaluate", *common, "--quick", "--data-label", "synthetic", cwd=tmp_path)
    run("inventory_rl.report", *common, "--reports", str(reports), "--validation-episodes", "4", cwd=tmp_path)
    run("inventory_rl.animate", "--artifacts", str(artifacts), "--reports", str(reports), cwd=tmp_path)

    summary = pd.read_csv(artifacts / "evaluation" / "summary.csv")
    assert len(summary) == 3 * 4 + 2
    assert summary["test_fill_rate"].between(0, 1).all()
    assert (summary["test_cost"] > 0).all()
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    assert set(selection["products"]) == {"fast", "medium", "slow"}
    assert len(list((reports / "figures").glob("*.png"))) == 7
    assert (reports / "figures" / "simulated_year.gif").stat().st_size > 10_000
    text = (reports / "results.md").read_text()
    assert "## Reward shaping" in text and "## Validation against the exact optimum" in text
