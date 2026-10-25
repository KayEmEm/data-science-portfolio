"""Small synthetic VRPTW instances in the VRPLIB format of the Solomon files, for tests and ``make smoke``.

The names repeat those of the benchmark (the week R101 to R107 and one instance of each other
class), so that every stage runs unchanged. Customers are uniform (R), clustered (C) or both (RC);
each time window is placed so that a vehicle that serves only that customer is on time. The
reference solution in each ``.sol`` file is the first solution of the cheapest insertion heuristic,
not a best known solution: numbers from the smoke run say nothing about the benchmark.
"""

import argparse
from pathlib import Path

import numpy as np
import vrplib

from delivery_planning.config import SEED, WEEK

NAMES = [*WEEK, "C101", "C201", "R201", "RC101", "RC201"]
# Per class type: horizon, capacity, window width range.
TYPES = {"1": (230, 200, (20, 60)), "2": (1000, 700, (100, 300))}


def coordinates(rng: np.random.Generator, letters: str, n: int) -> np.ndarray:
    uniform = rng.integers(0, 101, size=(n, 2))
    centers = rng.integers(15, 86, size=(4, 2))
    clustered = np.clip(centers[rng.integers(0, 4, size=n)] + rng.integers(-8, 9, size=(n, 2)), 0, 100)
    if letters == "R":
        return uniform
    if letters == "C":
        return clustered
    return np.where(rng.random((n, 1)) < 0.5, uniform, clustered)


def make_instance(name: str, customers: int, rng: np.random.Generator) -> dict:
    letters = name.rstrip("0123456789")
    horizon, capacity, (low, high) = TYPES[name[len(letters)]]
    service = 90 if letters == "C" else 10
    if letters == "C":
        horizon *= 3
    coords = np.vstack([[50, 50], coordinates(rng, letters, customers)]).astype(np.int64)
    distance = np.sqrt(((coords - coords[0]) ** 2).sum(axis=1))
    windows = [[0, horizon]]
    for d in distance[1:]:
        earliest, latest = int(np.ceil(d)) + 1, int(np.floor(horizon - service - d)) - 1
        center = rng.uniform(earliest, latest)
        width = rng.uniform(low, high)
        ready = int(np.clip(np.floor(center - width / 2), 0, latest))
        due = int(np.clip(np.ceil(center + width / 2), max(earliest, ready + 1), latest))
        windows.append([ready, due])
    return {
        "NAME": name,
        "TYPE": "CVRPTW",
        "DIMENSION": customers + 1,
        "VEHICLES": customers,
        "CAPACITY": capacity,
        "SERVICE_TIME": service,
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        "NODE_COORD_SECTION": coords,
        "DEMAND_SECTION": np.concatenate([[0], rng.integers(1, 31, size=customers)]),
        "TIME_WINDOW_SECTION": np.array(windows, dtype=np.int64),
        "DEPOT_SECTION": np.array([1]),
    }


def write(folder: Path, customers: int = 25, seed: int = SEED) -> list[str]:
    from delivery_planning.data import read_instance
    from delivery_planning.routing import solve

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in NAMES:
        path = folder / f"{name}.vrp"
        vrplib.write_instance(path, make_instance(name, customers, rng))
        result = solve(read_instance(path), metaheuristic=None, seconds=None)
        vrplib.write_solution(folder / f"{name}.sol", result["routes"], {"Cost": result["distance"] / 10})
    return NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--customers", type=int, default=25)
    args = parser.parse_args()
    names = write(args.folder, args.customers)
    print(f"{len(names)} synthetic instances in {args.folder}")


if __name__ == "__main__":
    main()
