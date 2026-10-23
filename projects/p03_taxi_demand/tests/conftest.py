from pathlib import Path

import pytest

from taxi_demand.synthetic import write_raw
from taxi_demand.warehouse import build


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    write_raw(out, year=2024, months=3, trips_per_month=1500)
    return out


@pytest.fixture(scope="session")
def built(raw_dir, tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("built")
    warehouse, marts = root / "taxi.duckdb", root / "marts"
    build(raw_dir, warehouse, marts, year=2024)
    return {"warehouse": warehouse, "marts": marts, "root": root}
