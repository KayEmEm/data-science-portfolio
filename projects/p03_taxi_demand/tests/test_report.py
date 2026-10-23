import json

from taxi_demand import report


def test_report_writes_tables_numbers_and_figures(built, monkeypatch):
    out = built["root"] / "reports"
    monkeypatch.setattr("sys.argv", ["report", "--warehouse", str(built["warehouse"]), "--reports", str(out)])
    report.main()
    numbers = json.loads((out / "key_numbers.json").read_text())
    assert 0 < numbers["usable_share"] <= 1
    assert numbers["usable_trips"] <= numbers["source_rows"]
    assert (out / "query_results.md").read_text().count("## a") == 11
    for name in ["daily_demand", "hour_weekday", "weather_effect", "zone_concentration", "speed_by_hour"]:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
