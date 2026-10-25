from streamlit.testing.v1 import AppTest

from delivery_planning.config import PROJECT_DIR
from delivery_planning.data import copy_source

APP = str(PROJECT_DIR / "planner" / "app.py")


def test_planner_plans_routes_and_shifts(source, tmp_path, monkeypatch):
    copy_source(source, tmp_path / "raw")
    monkeypatch.setenv("P10_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert any("press Plan" in i.value for i in app.info)
    app.sidebar.slider[1].set_value(1)
    app.sidebar.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Distance" in labels and "Driver cost (shift model)" in labels
    assert len(app.dataframe) == 2


def test_planner_reports_a_missing_data_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("P10_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert any("make data" in e.value for e in app.error)
