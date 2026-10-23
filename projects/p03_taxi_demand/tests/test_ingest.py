import shutil

from taxi_demand.ingest import ingest, kind_of


def test_files_are_identified_by_content_and_renamed(raw_dir, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i, path in enumerate(sorted((raw_dir / "trips").glob("*.parquet"))):
        shutil.copyfile(path, inbox / f"upload_{i}")
    shutil.copyfile(raw_dir / "taxi_zone_lookup.csv", inbox / "a")
    shutil.copyfile(raw_dir / "weather_central_park.csv", inbox / "b")
    (inbox / "notes").write_text("unrelated", encoding="utf-8")

    out = tmp_path / "raw"
    placed = ingest([inbox], out)
    assert sorted(p.name for p in placed["trips"]) == [
        "yellow_tripdata_2024-01.parquet",
        "yellow_tripdata_2024-02.parquet",
        "yellow_tripdata_2024-03.parquet",
    ]
    assert (out / "taxi_zone_lookup.csv").exists()
    assert (out / "weather_central_park.csv").exists()
    assert [p.name for p in placed["other"]] == ["notes"]


def test_unrelated_parquet_is_other(tmp_path):
    import pandas as pd

    path = tmp_path / "x"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    assert kind_of(path) == "other"
