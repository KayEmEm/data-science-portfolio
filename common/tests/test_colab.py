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


def test_drive_folders_use_the_project_number_unless_only_the_full_name_exists(tmp_path):
    project = run_project.project_dir("p01")
    assert run_project.drive_name(tmp_path, project) == "p01"
    (tmp_path / "raw" / "p01_loan_default").mkdir(parents=True)
    assert run_project.drive_name(tmp_path, project) == "p01_loan_default"
    (tmp_path / "raw" / "p01").mkdir()
    assert run_project.drive_name(tmp_path, project) == "p01"


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


def test_check_raw_takes_each_kind_of_data_from_its_subfolder_whatever_the_names(tmp_path):
    entries = run_project.read_raw_files(run_project.project_dir("p11"))
    for folder in ("features", "classes", "edges"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"my {folder} copy.csv").write_text(folder)
    assert run_project.check_raw(tmp_path, entries) == []
    assert (tmp_path / "elliptic_txs_edgelist.csv").read_text() == "edges"
    assert not any(list((tmp_path / f).iterdir()) for f in ("features", "classes", "edges"))

    p06 = tmp_path / "p06"
    for folder, name in (("czech", "RDD2022 CZ.zip"), ("india", "india data.zip")):
        _zip(p06 / folder / name, {"x.xml": "x"})
    assert (
        run_project.check_raw(p06, run_project.read_raw_files(run_project.project_dir("p06"))) == []
    )
    assert (p06 / "Czech.zip").is_file() and (p06 / "India.zip").is_file()


def test_subfolder_zip_gives_the_expected_member_or_the_only_one_of_its_kind(tmp_path):
    p02 = run_project.read_raw_files(run_project.project_dir("p02"))
    root = "financebench-cc39/"
    members = {
        root + "data/financebench_open_source.jsonl": "q",
        root + "data/financebench_document_information.jsonl": "d",
        **{f"{root}pdfs/f{i}.pdf": "p" for i in range(84)},
    }
    _zip(tmp_path / "questions" / "financebench.zip", members)
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "docs.jsonl").write_text("d")
    assert run_project.check_raw(tmp_path, p02[:2]) == []
    assert (tmp_path / "financebench_open_source.jsonl").read_text() == "q"

    p09 = tmp_path / "p09"
    _zip(p09 / "transactions" / "online+retail+ii.zip", {"retail 2009-2011.xlsx": "x"})
    assert (
        run_project.check_raw(p09, run_project.read_raw_files(run_project.project_dir("p09"))) == []
    )
    assert (p09 / "online_retail_II.xlsx").is_file()


def test_subfolder_for_many_files_and_too_many_candidates(tmp_path):
    p10 = run_project.read_raw_files(run_project.project_dir("p10"))
    members = {f"I/VRPTW/Solomon/C{i:03d}.{s}": "x" for i in range(56) for s in ("vrp", "sol")}
    _zip(tmp_path / "instances" / "Instances-7474.zip", members)
    assert run_project.check_raw(tmp_path, p10) == []
    assert len(list(tmp_path.glob("*.vrp"))) == 56

    p04 = tmp_path / "p04"
    (p04 / "customers").mkdir(parents=True)
    for name in ("a.csv.gz", "b.csv.gz"):
        (p04 / "customers" / name).write_text("x")
    missing = run_project.check_raw(p04, run_project.read_raw_files(run_project.project_dir("p04")))
    assert len(missing) == 1 and "found 2" in missing[0] and "a.csv.gz" in missing[0]
    (p04 / "customers" / "b.csv.gz").unlink()
    assert (
        run_project.check_raw(p04, run_project.read_raw_files(run_project.project_dir("p04"))) == []
    )
    assert (p04 / "hillstrom.csv.gz").is_file()


def test_missing_kind_names_its_subfolder(tmp_path):
    missing = run_project.check_raw(
        tmp_path, run_project.read_raw_files(run_project.project_dir("p07"))
    )
    assert missing[0].startswith("ratings/ (any file name)")


def test_p13_takes_the_full_records_or_else_the_partition(tmp_path):
    entries = run_project.read_raw_files(run_project.project_dir("p13"))
    for folder, name in (
        ("credit", "default+of+credit+card+clients.zip"),
        ("images", "czech.zip"),
        ("detector", "best.pt"),
    ):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_text("x")
    (tmp_path / "flows_records").mkdir()
    for i in range(1, 5):
        (tmp_path / "flows_records" / f"UNSW-NB15_{i}.csv").write_text("x")
    assert run_project.check_raw(tmp_path, entries) == []
    assert len(list((tmp_path / "flows").glob("UNSW-NB15_*.csv"))) == 4
    missing = run_project.check_raw(tmp_path / "none", entries)
    assert any("flows_records/" in line for line in missing)
    assert any("flows_train/" in line for line in missing)
