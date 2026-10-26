"""Validation against the exact optimum: dynamic programming on the slow product with a fixed lead time.

With one lead time L, an open order is known by the day it arrives, and the state on the morning
of day t is (on hand, units due in 1, ..., L - 1 days). For the slow product (capacity under 30
units) and L = 3 that is a few thousand states, so backward induction over the 365 days of the
episode gives the policy with the lowest expected yearly cost from the scored start. The
weekday of each day is known from t, so the weekday means of the demand model enter exactly.

Two checks follow:

1. the simulator, run with the optimal policy, must reproduce the expected cost of the
   induction within its sampling error (a check of the environment's dynamics and costs);
2. the tuned (s, S) and base stock policies, tuned the same way as in the main study but on
   this fixed lead time, are compared with the optimum: the gap shows how much any policy,
   learned or not, could gain over them in this setting;
3. when the PPO runs of this setting are trained (``train.py``, ``*_fixedlead_*``), the run
   with the lowest validation cost is scored the same way, and the gap of every seed is listed.

Output: ``artifacts/dp/result.json``.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from portfolio_common.config import parse_args
from scipy import stats

from inventory_rl import demand as demand_model
from inventory_rl.baselines import tune
from inventory_rl.config import ARTIFACTS_DIR, DATA_DIR, TEST_SEED, TEST_START, settings
from inventory_rl.data import load_demand, load_products
from inventory_rl.dynamics import Problem, base_stock, model_policy, s_s, seeded_scenarios, simulate

DP_PRODUCT = "slow"
DP_LEAD = 3
MAX_CAPACITY = 60  # the state space grows with the cube of the capacity


def leftover_matrix(mean: float, dispersion: float, cap: int) -> np.ndarray:
    """P[x, y]: probability that y units are left after the demand of one day, starting from x."""
    n, p = demand_model.nb_params(np.array([mean]), dispersion)
    pmf = stats.nbinom.pmf(np.arange(cap + 1), n, p[0])
    matrix = np.zeros((cap + 1, cap + 1))
    for x in range(cap + 1):
        matrix[x, 1 : x + 1] = pmf[:x][::-1]  # y = x - d for d = x - 1 ... 0
        matrix[x, 0] = 1.0 - matrix[x, 1:].sum()  # demand of x or more
    return matrix


def day_cost(problem: Problem, mean: float, cap: int) -> np.ndarray:
    """Expected holding and lost sales cost of a day that starts with x units on hand."""
    n, p = demand_model.nb_params(np.array([mean]), problem.dispersion)
    d = np.arange(0, 20 * cap + 200)
    pmf = stats.nbinom.pmf(d, n, p[0])
    x = np.arange(cap + 1)[:, None]
    left = np.maximum(x - d, 0)
    lost = np.maximum(d - x, 0)
    return (pmf * (problem.holding * left + problem.lost_sale * lost)).sum(axis=1)


def solve(problem: Problem, first_wday: int) -> tuple[np.ndarray, np.ndarray]:
    """Optimal order per day and state for lead time ``DP_LEAD``; returns (policy, value of day 0).

    Arrays are indexed [x, p_1, ..., p_{L-1}] with p_k the units due in k days.
    """
    if problem.lead_times != (DP_LEAD,):
        raise ValueError("the dynamic program needs the fixed lead time")
    cap = problem.capacity
    if cap > MAX_CAPACITY:
        raise ValueError(f"capacity {cap} is too large for the dynamic program")
    dims = (cap + 1,) * DP_LEAD  # x and the L - 1 pipeline slots
    grid = np.indices(dims)
    position = grid.sum(axis=0)
    orders = np.arange(cap + 1)
    value = np.zeros(dims)
    policy = np.zeros((problem.horizon, *dims), dtype="int16")
    for t in range(problem.horizon - 1, -1, -1):
        mean = problem.weekday_mean[(first_wday - 1 + t) % 7]
        leftover = leftover_matrix(mean, problem.dispersion, cap)
        immediate = day_cost(problem, mean, cap)
        q_values = np.full((cap + 1, *dims), np.inf)
        for q in orders:
            # Next morning: x' = leftover + p_1, the pipeline moves up one day, and q is due in L - 1 days.
            index = (
                [np.minimum(grid[0] + grid[1], cap)]
                + [grid[k] for k in range(2, DP_LEAD)]
                + [np.full(dims, q)]
            )
            nxt = value[tuple(index)]  # value of (y + p_1, p_2, ..., q) with y in the first axis
            expected = np.tensordot(leftover, nxt, axes=(1, 0))  # sum over the leftover y
            total = problem.order_cost * (q > 0) + immediate.reshape((-1,) + (1,) * (DP_LEAD - 1)) + expected
            q_values[q] = np.where(position + q <= cap, total, np.inf)
        policy[t] = q_values.argmin(axis=0)
        value = q_values.min(axis=0)
        # States above the capacity are never reached; a finite value keeps 0 x inf out of the sums.
        value = np.where(np.isfinite(value), value, 0.0)
    return policy, value


def dp_policy(policy: np.ndarray):
    def act(batch) -> np.ndarray:
        index = (
            np.full(batch.size, batch.t),
            batch.on_hand,
            *[batch.pipeline[:, k] for k in range(DP_LEAD - 1)],
        )
        return policy[index]

    return act


def run(product: dict, first_wday: int, episodes: int, tune_episodes: int, artifacts: Path) -> dict:
    problem = replace(Problem.from_product(product), lead_times=(DP_LEAD,), lead_probs=(1.0,))
    policy, value = solve(problem, first_wday)
    start = problem.initial_stock()
    expected = float(value[(start,) + (0,) * (DP_LEAD - 1)])
    scenarios = seeded_scenarios(problem, TEST_SEED, episodes, first_wday=first_wday)
    tuned, _ = tune(problem, tune_episodes)
    policies = {
        "optimal": dp_policy(policy),
        "s_S": s_s(tuned["s_S"]["s"], tuned["s_S"]["S"]),
        "base_stock": base_stock(tuned["base_stock"]["S"], tuned["base_stock"]["R"]),
    }
    out = {
        "product": problem.name,
        "lead_time": DP_LEAD,
        "capacity": problem.capacity,
        "expected_optimal": expected,
    }
    costs = {name: simulate(problem, scenarios, p)["totals"]["cost"] for name, p in policies.items()}
    learned = {}
    for path in sorted((artifacts / "runs").glob(f"{DP_PRODUCT}_ppo_fixedlead_s*/best.zip")):
        from stable_baselines3 import PPO

        cost = simulate(problem, scenarios, model_policy(PPO.load(path, device="cpu")))["totals"]["cost"]
        learned[path.parent.name] = float((cost - costs["optimal"]).mean() / costs["optimal"].mean())
        costs.setdefault("ppo_runs", []).append(cost)
    if learned:
        validation = {
            name: json.loads((artifacts / "runs" / name / "info.json").read_text())["best_validation_cost"]
            for name in learned
        }
        chosen = min(validation, key=validation.get)
        costs["ppo"] = costs.pop("ppo_runs")[list(learned).index(chosen)]
        tuned["ppo"] = {"run": chosen, "seed_gaps": learned}
    costs.pop("ppo_runs", None)
    for name, cost in costs.items():
        sd = float(cost.std(ddof=1))
        out[name] = {"mean": float(cost.mean()), "half_width": 1.96 * sd / np.sqrt(len(cost))}
        if name != "optimal":
            gap = cost - costs["optimal"]
            out[name] |= {
                "gap": float(gap.mean()),
                "gap_half_width": 1.96 * float(gap.std(ddof=1)) / np.sqrt(len(gap)),
                "gap_share": float(gap.mean() / costs["optimal"].mean()),
                "params": {k: v for k, v in tuned[name].items() if k != "tune_cost"},
            }
    out["episodes"] = episodes
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of the scenarios")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    args = parse_args(parser, __file__)
    run_settings = settings(args.sample, args.quick)
    product = {p["product"]: p for p in load_products(args.data_dir)}[DP_PRODUCT]
    demand = load_demand(args.data_dir)
    first_wday = int(demand[(demand["product"] == DP_PRODUCT) & (demand["d"] == TEST_START)]["wday"].iloc[0])
    result = run(
        product, first_wday, 4 * run_settings.test_episodes, run_settings.tune_episodes, args.artifacts
    )
    out = args.artifacts / "dp"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"{DP_PRODUCT}, lead time {DP_LEAD}: optimum {result['expected_optimal']:.2f} expected, "
        f"{result['optimal']['mean']:.2f} +- {result['optimal']['half_width']:.2f} simulated; "
        f"(s, S) +{100 * result['s_S']['gap_share']:.1f}%, "
        f"base stock +{100 * result['base_stock']['gap_share']:.1f}%"
    )


if __name__ == "__main__":
    main()
