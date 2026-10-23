from streamlit.testing.v1 import AppTest

from taxi_demand.config import PROJECT_DIR

APP = str(PROJECT_DIR / "dashboard" / "app.py")


def test_dashboard_renders_every_tab_without_error(built, monkeypatch):
    monkeypatch.setenv("TAXI_MARTS", str(built["marts"]))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Overview", "Time patterns", "Zones", "Weather", "Data quality"]


def test_dashboard_warns_when_no_borough_is_selected(built, monkeypatch):
    monkeypatch.setenv("TAXI_MARTS", str(built["marts"]))
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No trips match" in w.value for w in app.warning)
