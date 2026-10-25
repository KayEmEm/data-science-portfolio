from pathlib import Path

import pytest

from adv_robustness import data, synthetic


@pytest.fixture(scope="session")
def sources(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    synthetic.write(out, clients=800, flows=6000, images=16, image_size=64)
    return out


@pytest.fixture(scope="session")
def data_dir(sources, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    data.credit(sources / "credit", out / "credit", 1.0, 42)
    data.flows(sources / "flows", out / "flows", 1.0, 42)
    data.images(sources / "images", out / "images", 1.0, 42, block=2)
    return out
