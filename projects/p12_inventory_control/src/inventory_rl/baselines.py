"""Tune the classic policies by simulation: periodic review base stock (R, S) and (s, S).

Each policy is scored by its mean yearly cost on ``tune_episodes`` scenarios of the tuning seed
stream, which no other step uses. The search is a coarse grid over the capacity and then a fine
grid around the best coarse point; every grid point runs on the same scenarios. Results go to
``artifacts/baselines/<product>.json`` and the cost of every grid point to ``<product>_grid.csv``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_rl.config import ARTIFACTS_DIR, DATA_DIR, TUNE_SEED, settings
from inventory_rl.data import load_products
from inventory_rl.dynamics import Problem, Scenarios, base_stock, s_s, seeded_scenarios, simulate

CHUNK_EPISODES = 60_000  # episodes simulated at once (grid points x scenarios)
REVIEWS = (1, 2, 3, 7, 14, 28, 42, 56)  # review periods of the base stock policy, in days


def grid_costs(problem: Problem, scenarios: Scenarios, points: np.ndarray, make_policy) -> np.ndarray:
    """Mean yearly cost of each row of ``points`` (one policy per row) on all scenarios."""
    n = len(scenarios)
    per_chunk = max(1, CHUNK_EPISODES // n)
    means = []
    for start in range(0, len(points), per_chunk):
        block = points[start : start + per_chunk]
        repeated = scenarios.repeat(len(block))
        # repeat() puts the copies of one scenario next to each other: episode i runs point i % len(block).
        params = np.tile(block, (n, 1))
        cost = simulate(problem, repeated, make_policy(params))["totals"]["cost"]
        means.append(cost.reshape(n, len(block)).mean(axis=0))
    return np.concatenate(means)


def steps(capacity: int, parts: int) -> int:
    return max(1, capacity // parts)


def tune_base_stock(problem: Problem, scenarios: Scenarios) -> tuple[dict, pd.DataFrame]:
    """Order up to S every R days, R in ``REVIEWS``; a daily base stock pays the order cost every day."""
    cap = problem.capacity
    width = steps(cap, 60)

    def make(p):
        return base_stock(p[:, 1], p[:, 0])

    coarse = np.array([(r, s) for r in REVIEWS for s in range(0, cap + 1, width)])
    cost = grid_costs(problem, scenarios, coarse, make)
    review, best = coarse[cost.argmin()]
    fine = np.array([(review, s) for s in range(max(0, best - 2 * width), min(cap, best + 2 * width) + 1)])
    fine_cost = grid_costs(problem, scenarios, fine, make)
    grid = pd.DataFrame(np.vstack([coarse, fine]), columns=["R", "S"])
    grid["cost"] = np.r_[cost, fine_cost]
    grid = grid.drop_duplicates(["R", "S"]).sort_values(["R", "S"])
    top = grid.loc[grid["cost"].idxmin()]
    return {"R": int(top["R"]), "S": int(top["S"]), "tune_cost": float(top["cost"])}, grid


def pairs(s_values: np.ndarray, big_values: np.ndarray) -> np.ndarray:
    s, big = np.meshgrid(s_values, big_values, indexing="ij")
    keep = big > s
    return np.column_stack([s[keep], big[keep]])


def tune_s_s(problem: Problem, scenarios: Scenarios) -> tuple[dict, pd.DataFrame]:
    cap = problem.capacity
    width = steps(cap, 40)
    coarse = pairs(np.arange(0, cap, width), np.arange(width, cap + 1, width))

    def make(p):
        return s_s(p[:, 0], p[:, 1])

    cost = grid_costs(problem, scenarios, coarse, make)
    s0, big0 = coarse[cost.argmin()]
    fine_width = max(1, width // 6)
    fine = pairs(
        np.arange(max(0, s0 - 2 * width), min(cap - 1, s0 + 2 * width) + 1, fine_width),
        np.arange(max(1, big0 - 2 * width), min(cap, big0 + 2 * width) + 1, fine_width),
    )
    fine_cost = grid_costs(problem, scenarios, fine, make)
    grid = pd.DataFrame(np.vstack([coarse, fine]), columns=["s", "S"])
    grid["cost"] = np.r_[cost, fine_cost]
    grid = grid.drop_duplicates(["s", "S"])
    top = grid.loc[grid["cost"].idxmin()]
    return {"s": int(top["s"]), "S": int(top["S"]), "tune_cost": float(top["cost"])}, grid


def tune(problem: Problem, episodes: int) -> tuple[dict, dict[str, pd.DataFrame]]:
    scenarios = seeded_scenarios(problem, TUNE_SEED, episodes)
    base, base_grid = tune_base_stock(problem, scenarios)
    ss, ss_grid = tune_s_s(problem, scenarios)
    return {"base_stock": base, "s_S": ss, "tune_episodes": episodes}, {
        "base_stock": base_grid,
        "s_S": ss_grid,
    }


def load_baselines(artifacts: Path, product: str) -> dict:
    path = artifacts / "baselines" / f"{product}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run make baselines first")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of the tuning scenarios")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    args = parser.parse_args()
    run = settings(args.sample, args.quick)
    out = args.artifacts / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    for product in load_products(args.data_dir):
        problem = Problem.from_product(product)
        result, grids = tune(problem, run.tune_episodes)
        (out / f"{problem.name}.json").write_text(json.dumps(result, indent=2))
        for name, grid in grids.items():
            grid.to_csv(out / f"{problem.name}_{name}_grid.csv", index=False)
        b, s = result["base_stock"], result["s_S"]
        print(
            f"{problem.name}: capacity {problem.capacity}, "
            f"base stock (R, S)=({b['R']}, {b['S']}) ({b['tune_cost']:.0f} per year), "
            f"(s, S)=({s['s']}, {s['S']}) ({s['tune_cost']:.0f} per year)"
        )


if __name__ == "__main__":
    main()
