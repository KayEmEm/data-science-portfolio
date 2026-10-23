import json

import pandas as pd

from retail_forecast import backtest, evaluate, report
from retail_forecast.demand import daily_demand


def run(module, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    module.main()


def test_demand_tables_for_reuse(data_dir, tables):
    level9 = daily_demand(9, data_dir)
    assert {"level", "series", "store_id", "dept_id", "d", "date", "units"} <= set(level9.columns)
    assert level9["series"].nunique() == 9
    assert level9["units"].sum() == tables["sales"].sum()
    bottom = daily_demand(12, data_dir)
    assert "sell_price" in bottom.columns and len(bottom) == 54 * 500


def test_end_to_end_on_synthetic_data(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    artifacts = tmp_path / "artifacts"
    common = ["--data-dir", str(data_dir), "--artifacts", str(artifacts)]
    run(backtest, [*common, "--quick", "--models", "snaive", "ets", "prophet", "lgbm"], monkeypatch)
    run(evaluate, [*common, "--quick", "--data-label", "synthetic"], monkeypatch)
    run(report, [*common, "--reports", str(tmp_path / "reports")], monkeypatch)

    summary = pd.read_csv(artifacts / "evaluation" / "summary.csv", index_col="method")
    expected = {
        "snaive",
        "ets",
        "ets_ols",
        "ets_wls_struct",
        "ets_mint_shrink",
        "prophet",
        "lgbm",
        "ensemble",
    }
    assert set(summary.index) == expected
    # The synthetic demand has a weekly pattern, a trend and price effects that
    # every model sees; the learned models must beat repeating the last week.
    assert summary.loc["lgbm", "backtest_wrmsse"] < summary.loc["snaive", "backtest_wrmsse"]
    assert summary.loc["ets", "backtest_wrmsse"] < summary.loc["snaive", "backtest_wrmsse"]

    intervals = pd.read_csv(artifacts / "evaluation" / "intervals.csv")
    assert set(intervals["kind"]) == {"native", "conformal"}
    assert intervals["coverage"].between(0, 1).all()
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    assert selection["selected"] == summary.index[0]
    results = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert results["data"] == "synthetic"
    figures = {p.name for p in (tmp_path / "reports" / "figures").glob("*.png")}
    assert figures == {
        "wrmsse_by_model.png",
        "wrmsse_by_level.png",
        "test_forecasts.png",
        "interval_coverage.png",
        "lgbm_importance.png",
    }
