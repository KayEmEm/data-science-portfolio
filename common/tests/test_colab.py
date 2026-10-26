import importlib.util
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_project", REPO / "colab" / "run_project.py")
run_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_project)


def test_local_packages_follow_the_install_target():
    names = [p.name for p in run_project.local_packages(run_project.project_dir("p13"))]
    assert names == [
        "common",
        "p01_loan_default",
        "p06_road_damage",
        "p08_network_anomaly",
        "p13_adversarial_robustness",
    ]


def test_requirements_leave_out_local_and_colab_packages():
    project = run_project.project_dir("p13")
    specs = run_project.requirements(run_project.local_packages(project))
    names = {s.split("=")[0].lower() for s in specs}
    assert "numpy" in names and "pyyaml" in names
    assert not names & {"torch", "torchvision", "portfolio-common", "loan-default"}
    assert all("==" in s for s in specs)


@pytest.mark.parametrize("name", ["free", "pro"])
def test_profiles_are_valid_for_every_project(name):
    profile = run_project.read_profile(name)
    for project in sorted((REPO / "projects").glob("p[0-9][0-9]_*")):
        values = run_project.for_project(profile, project)
        targets = values["target"].split()
        makefile = (project / "Makefile").read_text()
        assert all(f"\n{t}:" in makefile for t in targets), (project.name, targets)


def test_project_entry_overrides_the_profile():
    profile = {
        "target": "sample",
        "sample": None,
        "projects": {"p12": {"target": "all", "sample": 0.1}},
    }
    values = run_project.for_project(profile, run_project.project_dir("p12"))
    assert values == {"target": "all", "sample": 0.1}
    assert run_project.for_project(profile, run_project.project_dir("p01"))["target"] == "sample"


def test_copy_tree_copies_changed_files_only_and_skips_excluded(tmp_path):
    source, target = tmp_path / "a", tmp_path / "b"
    (source / "data").mkdir(parents=True)
    (source / "data" / "big.bin").write_text("x")
    (source / "m.json").write_text("1")
    assert run_project.copy_tree(source, target, exclude=("data",)) == 1
    assert not (target / "data").exists()
    assert run_project.copy_tree(source, target, exclude=("data",)) == 0
    time.sleep(0.01)
    (source / "m.json").write_text("22")
    os.utime(source / "m.json")
    assert run_project.copy_tree(source, target) == 2
    assert run_project.copy_tree(tmp_path / "none", target) == 0


def test_make_command_splits_targets_and_sets_sample():
    command = run_project.make_command(Path("/p"), "data sample", 0.2)
    assert command[:5] == ["make", "-C", "/p", "data", "sample"]
    assert command[-1] == "SAMPLE=0.2"
