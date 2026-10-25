import json
import re
import zipfile

import numpy as np
import pytest

from delivery_planning import data
from delivery_planning.data import (
    SchemaError,
    audit,
    check_solution,
    copy_source,
    instance_class,
    schedule,
    truncated_distances,
)


def test_distances_are_truncated_to_one_decimal_exactly():
    coords = np.array([[0, 0], [3, 4], [1, 1], [10, 0]])
    dist = truncated_distances(coords)
    assert dist[0, 1] == 50  # 5.0 exactly, no rounding down to 49
    assert dist[0, 2] == 14  # 1.414 truncates to 1.4
    assert dist[1, 3] == 80  # sqrt(65) = 8.06
    assert (dist == dist.T).all() and (np.diag(dist) == 0).all()


def test_instance_class_from_name():
    assert [instance_class(n) for n in ["C101", "C208", "R112", "RC201"]] == ["C1", "C2", "R1", "RC2"]


def test_audit_recomputes_every_reference_solution(source, tmp_path):
    raw = tmp_path / "raw"
    copy_source(source, raw)
    table = audit(raw)
    assert len(table) == 12
    assert set(table["class"]) == {"C1", "C2", "R1", "R2", "RC1", "RC2"}
    assert (table["customers"] == 20).all()


def test_audit_rejects_a_wrong_stated_cost(source, tmp_path):
    raw = tmp_path / "raw"
    copy_source(source, raw)
    path = raw / "R101.sol"
    path.write_text(
        re.sub(r"Cost: ([\d.]+)", lambda m: f"Cost: {float(m.group(1)) + 1:.1f}", path.read_text())
    )
    with pytest.raises(SchemaError, match="recomputed distance"):
        audit(raw)


def test_copy_source_reads_a_zip(source, tmp_path):
    archive = tmp_path / "instances.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for path in source.iterdir():
            out.write(path, f"nested/{path.name}")
    copy_source(archive, tmp_path / "raw")
    assert len(list((tmp_path / "raw").glob("*.vrp"))) == 12


def test_verify_detects_a_changed_file(source, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    copy_source(source, raw)
    expected = {p.name: data.sha256(p) for p in raw.iterdir()}
    monkeypatch.setattr(data, "manifest", lambda: expected)
    data.verify(raw)
    (raw / "C101.vrp").write_text((raw / "C101.vrp").read_text() + "\n")
    with pytest.raises(SchemaError, match="SHA256"):
        data.verify(raw)
    monkeypatch.setattr(data, "manifest", lambda: {**expected, "X1.vrp": "0"})
    with pytest.raises(SchemaError, match="missing"):
        data.verify(raw)


def test_check_solution_finds_each_kind_of_violation(r101):
    instance, best = r101
    assert check_solution(instance, best["routes"])["problems"] == []
    routes = [list(r) for r in best["routes"]]
    assert "a customer is visited twice" in check_solution(instance, [*routes, [routes[0][0]]])["problems"]
    assert any("not visited" in p for p in check_solution(instance, routes[1:])["problems"])
    everyone = [c for r in routes for c in r]
    problems = check_solution(instance, [everyone])["problems"]
    assert any("over capacity" in p for p in problems) or any("late" in p for p in problems)


def test_schedule_waits_for_windows_and_departs_as_late_as_possible(r101):
    instance, best = r101
    for route in best["routes"]:
        times = schedule(instance, route)
        assert all(t >= instance.ready[c] for c, t in zip(route, times["starts"], strict=True))
        assert times["depart"] + instance.dist[0, route[0]] <= times["starts"][0]
        # Departing at ``depart`` still meets every window and returns at the same time.
        clock, prev = times["depart"], 0
        for node in route:
            clock = max(clock + instance.service[prev] + instance.dist[prev, node], instance.ready[node])
            assert clock <= instance.due[node]
            prev = node
        assert clock + instance.service[prev] + instance.dist[prev, 0] == times["ret"]


def test_manifest_pins_all_56_instances_and_solutions():
    names = json.loads(data.MANIFEST_PATH.read_text())
    assert len(names) == 112
    assert sum(n.endswith(".vrp") for n in names) == 56
