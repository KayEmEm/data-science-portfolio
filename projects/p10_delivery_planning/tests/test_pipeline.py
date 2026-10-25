import json
import os
import subprocess
import sys

import pandas as pd


def run(*args, cwd):
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=cwd, env=env)


def test_end_to_end_on_synthetic_instances(source, tmp_path):
    data, artifacts, reports = tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports"
    run(
        "delivery_planning.data",
        "--source",
        str(source),
        "--data-dir",
        str(data),
        "--sha",
        "any",
        cwd=tmp_path,
    )
    common = ["--data-dir", str(data), "--artifacts", str(artifacts), "--quick"]
    run("delivery_planning.train", *common, cwd=tmp_path)
    stored = artifacts / "benchmark" / "gls_10s" / "R101.json"
    stamp = stored.stat().st_mtime_ns
    # A second run finds every search stored and runs nothing again.
    run("delivery_planning.train", *common, cwd=tmp_path)
    assert stored.stat().st_mtime_ns == stamp
    run("delivery_planning.evaluate", *common, cwd=tmp_path)
    run("delivery_planning.report", "--data-dir", str(data), "--artifacts", str(artifacts),
        "--reports", str(reports), "--data-label", "synthetic", cwd=tmp_path)  # fmt: skip

    summary = pd.read_csv(artifacts / "evaluation" / "benchmark_summary.csv")
    assert set(summary["config"]) == {"cheapest_arc", "gls_10s"}
    assert (summary.loc[summary["class"] == "all", "found"] == 12).all()
    comparison = pd.read_csv(artifacts / "evaluation" / "solver_comparison.csv")
    solved = comparison.loc[comparison["cost"].notna()]
    # HiGHS and CBC always run; every solver that solves the small scenario reaches the same optimum.
    assert {"highs", "cbc"} <= set(solved["solver"])
    small = solved.loc[solved["scenario"] == "small"]
    assert small["cost"].max() - small["cost"].min() < 1e-6
    assert (solved["cost"] <= solved["greedy_cost"] + 1e-6).all()
    fleet = pd.read_csv(artifacts / "evaluation" / "fleet.csv")
    assert set(fleet["objective"]) == {"distance", "cost_aware", "benchmark_plan"}
    tightest = fleet.loc[fleet["cap"] == fleet["cap"].min()]
    assert (tightest["unserved"] > 0).all()
    roster = pd.read_csv(artifacts / "evaluation" / "roster.csv")
    assert len(roster) == len(pd.read_csv(artifacts / "evaluation" / "duties.csv"))
    assert len(list((reports / "figures").glob("*.png"))) == 5  # the time figure needs two time limits
    assert "leaflet" in (reports / "maps" / "R101.html").read_text()
    results = json.loads((reports / "results.json").read_text())
    assert results["data"] == "synthetic" and results["instances"] == 12
    assert "## Fleet size study" in (reports / "results.md").read_text()
