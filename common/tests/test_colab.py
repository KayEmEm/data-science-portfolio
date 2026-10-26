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


def _zip(path: Path, members: dict[str, str]) -> Path:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        for name, text in members.items():
            handle.writestr(name, text)
    return path


def test_raw_files_cover_every_project_with_known_keys():
    for project in sorted((REPO / "projects").glob("p[0-9][0-9]_*")):
        entries = run_project.read_raw_files(project)
        assert entries, project.name
        stack = list(entries)
        while stack:
            entry = stack.pop()
            if "one_of" in entry:
                stack += [item for option in entry["one_of"] for item in option]
                continue
            assert "file" in entry and "source" in entry, (project.name, entry)
            assert set(entry) <= run_project.ENTRY_KEYS, (project.name, entry)


def test_check_raw_renames_moves_and_extracts_for_the_real_entries(tmp_path):
    p01 = tmp_path / "p01"
    _zip(p01 / "default+of+credit+card+clients.zip", {"default of credit card clients.xls": "x"})
    assert (
        run_project.check_raw(p01, run_project.read_raw_files(run_project.project_dir("p01"))) == []
    )
    assert (p01 / "default_of_credit_card_clients.zip").is_file()

    p11 = tmp_path / "p11"
    names = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
    _zip(p11 / "archive.zip", {f"elliptic_bitcoin_dataset/{n}": n for n in names})
    assert (
        run_project.check_raw(p11, run_project.read_raw_files(run_project.project_dir("p11"))) == []
    )
    assert all((p11 / n).read_text() == n for n in names)

    p13 = tmp_path / "p13"
    p13.mkdir()
    for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv", "best.pt"):
        (p13 / name).write_text("x")
    _zip(p13 / "Czech.zip", {"Czech/train/a.xml": "x"})
    _zip(p13 / "default_of_credit_card_clients.zip", {"a.xls": "x"})
    assert (
        run_project.check_raw(p13, run_project.read_raw_files(run_project.project_dir("p13"))) == []
    )
    assert (p13 / "flows" / "UNSW_NB15_testing-set.csv").is_file()
    assert (p13 / "images" / "Czech.zip").is_file()
    assert (p13 / "credit" / "default_of_credit_card_clients.zip").is_file()
    assert (p13 / "detector" / "best.pt").is_file()


def test_check_raw_extracts_only_members_inside_the_given_path(tmp_path):
    entry = {"file": "*.vrp", "min": 2, "inside": "*Solomon/*", "source": "s"}
    members = {f"I/VRPTW/Solomon/C10{i}.vrp": "x" for i in (1, 2)}
    _zip(tmp_path / "instances.zip", {**members, "I/VRPTW/Homberger/C1_2_1.vrp": "x"})
    assert run_project.check_raw(tmp_path, [entry]) == []
    assert sorted(p.name for p in tmp_path.glob("*.vrp")) == ["C101.vrp", "C102.vrp"]


def test_check_raw_reports_missing_files_and_alternatives(tmp_path):
    entries = [
        {"file": "a.csv", "source": "https://a"},
        {"file": "z.zip", "extract": False, "source": "https://z"},
        {
            "one_of": [
                [{"file": "f[1-2].csv", "min": 2, "source": "full"}],
                [{"file": "p.csv", "source": "part"}],
            ]
        },
        {"one_of": [[{"file": "opt.csv", "source": "o"}], []]},
    ]
    _zip(tmp_path / "other.zip", {"z.zip": "x"})
    missing = run_project.check_raw(tmp_path, entries)
    assert missing[0].startswith("a.csv") and "https://a" in missing[0]
    assert missing[1].startswith("z.zip")
    assert missing[2] == "one of these sets:" and "found 0" in missing[3] and "part" in missing[4]
    assert len(missing) == 5
    (tmp_path / "p.csv").write_text("x")
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "z.zip").write_text("x")
    assert run_project.check_raw(tmp_path, entries) == []


def test_report_missing_stops_and_keeps_the_list_on_drive(tmp_path):
    project = run_project.project_dir("p04")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as stop:
        run_project.report_missing(
            project, ["hillstrom.csv.gz\n      from: https://x"], tmp_path, out
        )
    assert stop.value.code == 2
    assert "hillstrom.csv.gz" in (out / run_project.MISSING_NAME).read_text()
    run_project.report_missing(project, [], tmp_path, out)
    assert not (out / run_project.MISSING_NAME).exists()
