"""Download the Solomon VRPTW instances and best known solutions, check them, and read them.

The files come from the PyVRP instance collection on GitHub, pinned to one commit, and each is
checked against its SHA256 in ``solomon_files.json``. ``--source`` reads them from a local folder or
zip instead; ``--sha any`` skips the check (for the synthetic instances of the smoke run).

Distances follow the DIMACS convention of the best known solutions: Euclidean distance truncated to
one decimal. Coordinates are integers, so the truncation is exact in integer arithmetic. Times use
the same unit, so travel time equals distance.
"""

import argparse
import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import vrplib

from delivery_planning.config import CLASSES, DATA_DIR, MANIFEST_PATH, RAW_DIR, SCALE, SOURCE_URL


class SchemaError(ValueError):
    """A file does not match the documented layout."""


@dataclass
class Instance:
    """One VRPTW instance; index 0 is the depot. Distances and times in tenths of a unit."""

    name: str
    coords: np.ndarray  # integer coordinates, shape (n + 1, 2)
    demand: np.ndarray
    ready: np.ndarray  # earliest service start
    due: np.ndarray  # latest service start; due[0] is the latest return to the depot
    service: np.ndarray
    capacity: int
    vehicles: int
    dist: np.ndarray  # truncated Euclidean distance

    @property
    def customers(self) -> int:
        return len(self.demand) - 1

    @property
    def horizon(self) -> int:
        return int(self.due[0])

    @property
    def family(self) -> str:
        return instance_class(self.name)


def instance_class(name: str) -> str:
    """C1, C2, R1, R2, RC1 or RC2 from a name such as RC104."""
    letters = name.rstrip("0123456789")
    return f"{letters}{name[len(letters)]}"


def manifest() -> dict[str, str]:
    return json.loads(MANIFEST_PATH.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: Path = RAW_DIR) -> None:
    """Fetch every file of the manifest that is not present yet (112 files, about 0.6 MB)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in manifest():
        target = raw_dir / name
        if target.exists():
            continue
        with urllib.request.urlopen(SOURCE_URL.format(name=name), timeout=60) as response:
            target.write_bytes(response.read())


def copy_source(source: Path, raw_dir: Path) -> None:
    """Copy the .vrp and .sol files of a folder (searched recursively) or a zip into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name.endswith((".vrp", ".sol")):
                    (raw_dir / name).write_bytes(archive.read(member))
        return
    if not source.is_dir():
        raise SchemaError(f"{source} is neither a folder nor a zip")
    for path in sorted(source.rglob("*")):
        if path.suffix in (".vrp", ".sol") and path.resolve() != (raw_dir / path.name).resolve():
            shutil.copyfile(path, raw_dir / path.name)


def verify(raw_dir: Path) -> None:
    expected = manifest()
    missing = sorted(name for name in expected if not (raw_dir / name).exists())
    if missing:
        raise SchemaError(f"missing files: {missing[:5]} and {max(len(missing) - 5, 0)} more")
    wrong = sorted(name for name, digest in expected.items() if sha256(raw_dir / name) != digest)
    if wrong:
        raise SchemaError(f"SHA256 differs from the pinned commit: {wrong}")


def truncated_distances(coords: np.ndarray) -> np.ndarray:
    """floor(SCALE * Euclidean distance), exact for integer coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    squared = (diff**2).sum(axis=-1) * SCALE * SCALE
    return np.vectorize(math.isqrt, otypes=[np.int64])(squared)


def read_instance(path: Path) -> Instance:
    raw = vrplib.read_instance(str(path))
    for key in ("node_coord", "demand", "time_window", "service_time", "capacity", "depot"):
        if key not in raw:
            raise SchemaError(f"{path.name} lacks {key}")
    coords = np.asarray(raw["node_coord"])
    if not np.allclose(coords, np.round(coords)):
        raise SchemaError(f"{path.name} has coordinates that are not integers")
    if list(np.asarray(raw["depot"]).ravel()) != [0]:
        raise SchemaError(f"{path.name}: the depot is not the first node")
    coords = np.round(coords).astype(np.int64)
    n = len(coords)
    service = np.broadcast_to(np.asarray(raw["service_time"]), (n,)).astype(np.int64).copy()
    service[0] = 0
    window = np.asarray(raw["time_window"]).astype(np.int64)
    demand = np.asarray(raw["demand"]).astype(np.int64)
    if demand[0] != 0 or (demand[1:] <= 0).any():
        raise SchemaError(f"{path.name}: demands must be zero at the depot and positive elsewhere")
    return Instance(
        name=str(raw.get("name", path.stem)),
        coords=coords,
        demand=demand,
        ready=window[:, 0] * SCALE,
        due=window[:, 1] * SCALE,
        service=service * SCALE,
        capacity=int(raw["capacity"]),
        vehicles=int(raw.get("vehicles", n - 1)),
        dist=truncated_distances(coords),
    )


def read_solution(path: Path) -> dict:
    """Routes as lists of customer indices (1 to n) and the stated cost."""
    raw = vrplib.read_solution(str(path))
    return {"routes": [list(map(int, r)) for r in raw["routes"]], "cost": float(raw["cost"])}


def schedule(instance: Instance, route: list[int]) -> dict:
    """Earliest service starts along a route, and the shortest duty that serves it.

    ``depart`` is the latest departure from the depot that still returns at the earliest possible
    return time, so ``ret - depart`` is the time a driver needs for the route.
    """
    starts, time, prev = [], 0, 0
    for node in route:
        time = max(time + instance.service[prev] + instance.dist[prev, node], instance.ready[node])
        starts.append(int(time))
        prev = node
    ret = int(time + instance.service[prev] + instance.dist[prev, 0]) if route else 0
    latest = ret
    nxt = 0
    for node in reversed(route):
        latest = min(int(instance.due[node]), latest - int(instance.service[node] + instance.dist[node, nxt]))
        nxt = node
    depart = latest - int(instance.dist[0, route[0]]) if route else 0
    return {"starts": starts, "ret": ret, "depart": max(depart, int(instance.ready[0]))}


def check_solution(instance: Instance, routes: list[list[int]], require_all: bool = True) -> dict:
    """Distance, vehicles and the list of rule violations of a solution."""
    problems = []
    seen = [c for r in routes for c in r]
    if len(seen) != len(set(seen)):
        problems.append("a customer is visited twice")
    missing = set(range(1, instance.customers + 1)) - set(seen)
    if require_all and missing:
        problems.append(f"{len(missing)} customers not visited")
    if set(seen) - set(range(1, instance.customers + 1)):
        problems.append("unknown customer")
        return {"distance": None, "vehicles": len(routes), "served": len(set(seen)), "problems": problems}
    used = [r for r in routes if r]
    if len(used) > instance.vehicles:
        problems.append(f"{len(used)} routes for {instance.vehicles} vehicles")
    distance = 0
    for i, route in enumerate(used):
        path = [0, *route, 0]
        distance += int(instance.dist[path[:-1], path[1:]].sum())
        if instance.demand[route].sum() > instance.capacity:
            problems.append(f"route {i + 1} over capacity")
        times = schedule(instance, route)
        late = [c for c, t in zip(route, times["starts"], strict=True) if t > instance.due[c]]
        if late:
            problems.append(f"route {i + 1} late at {late}")
        if times["ret"] > instance.due[0]:
            problems.append(f"route {i + 1} returns after the depot closes")
    return {"distance": distance, "vehicles": len(used), "served": len(set(seen)), "problems": problems}


def instance_names(raw_dir: Path) -> list[str]:
    return sorted(p.stem for p in raw_dir.glob("*.vrp"))


def load(raw_dir: Path, name: str) -> tuple[Instance, dict]:
    return read_instance(raw_dir / f"{name}.vrp"), read_solution(raw_dir / f"{name}.sol")


def audit(raw_dir: Path) -> pd.DataFrame:
    """One row per instance; every best known solution is checked and its cost recomputed."""
    rows = []
    for name in instance_names(raw_dir):
        if not (raw_dir / f"{name}.sol").exists():
            raise SchemaError(f"{name} has no best known solution")
        instance, best = load(raw_dir, name)
        result = check_solution(instance, best["routes"])
        if result["problems"]:
            raise SchemaError(f"{name}: best known solution breaks a rule: {result['problems']}")
        stated = round(best["cost"] * SCALE)
        if result["distance"] != stated:
            raise SchemaError(f"{name}: recomputed distance {result['distance']} differs from {stated}")
        rows.append(
            {
                "instance": name,
                "class": instance.family,
                "customers": instance.customers,
                "capacity": instance.capacity,
                "vehicles_available": instance.vehicles,
                "horizon": instance.horizon / SCALE,
                "mean_window": float((instance.due[1:] - instance.ready[1:]).mean() / SCALE),
                "bks_distance": best["cost"],
                "bks_vehicles": result["vehicles"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="folder or zip with the .vrp and .sol files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sha", choices=["pinned", "any"], default="pinned")
    args = parser.parse_args()
    raw_dir = args.data_dir / "raw"
    if args.source:
        copy_source(args.source, raw_dir)
    elif args.sha == "pinned":
        download(raw_dir)
    if args.sha == "pinned":
        verify(raw_dir)
    table = audit(raw_dir)
    table.to_csv(args.data_dir / "instances.csv", index=False)
    summary = {
        "instances": len(table),
        "per_class": {c: int((table["class"] == c).sum()) for c in CLASSES if (table["class"] == c).any()},
        "customers": sorted(table["customers"].unique().tolist()),
        "checked": "every best known solution is feasible and its recomputed distance equals the stated cost",
        "sha256": args.sha,
    }
    (args.data_dir / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
