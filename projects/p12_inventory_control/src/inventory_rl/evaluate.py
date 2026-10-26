"""Score every policy on the same test scenarios and on a replay of the real test year.

Per product:

* test: ``test_episodes`` simulated years of the fitted demand (test seed stream, never used
  in tuning or training), all starting on the weekday of the first test day with the same stock;
* replay: the real M5 units of the test year (d_1605 to d_1969), with ``replay_episodes`` draws
  of the lead times. The demand model never saw these days, so this checks how the policies
  carry over to demand that is not drawn from their own model.

Policies: the tuned base stock (R, S) and (s, S), and the best validation checkpoint of every RL
run. For each RL algorithm the seed with the lowest validation cost is the selected policy; the
spread over seeds is reported too. Differences against (s, S) are paired on the scenarios.
Outputs go to ``artifacts/evaluation/``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from scipy import stats

from inventory_rl.baselines import load_baselines
from inventory_rl.config import (
    ALGORITHMS,
    ARTIFACTS_DIR,
    DATA_DIR,
    REPLAY_SEED,
    TEST_SEED,
    TEST_START,
    settings,
)
from inventory_rl.data import load_demand, load_products
from inventory_rl.dynamics import (
    COST_PARTS,
    Problem,
    base_stock,
    model_policy,
    s_s,
    seeded_scenarios,
    simulate,
)
from inventory_rl.train import EXPERIMENT, planned_runs

REFERENCE = "s_S"
TRACE_EPISODES = 1


def mean_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype="float64")
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf((1 + level) / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def summarize(totals: dict, reference: np.ndarray | None) -> dict:
    mean, low, high = mean_interval(totals["cost"])
    row = {"cost": mean, "cost_low": low, "cost_high": high}
    row |= {k: float(np.mean(totals[k])) for k in (*COST_PARTS, "fill_rate", "orders", "on_hand", "lost")}
    if reference is not None:
        diff, d_low, d_high = mean_interval(totals["cost"] - reference)
        row |= {
            "diff": diff,
            "diff_low": d_low,
            "diff_high": d_high,
            "diff_share": diff / float(reference.mean()),
        }
    return row


def load_model(path: Path, algorithm: str):
    from stable_baselines3 import PPO, SAC

    return (PPO if algorithm == "ppo" else SAC).load(path, device="cpu")


def policy_map(problem: Problem, policies: dict, wday: int = 3) -> pd.DataFrame:  # 3: Monday
    """Order placed by each policy at every position, with all stock on hand and nothing on order."""
    from inventory_rl.dynamics import Batch, Scenarios

    positions = np.arange(problem.capacity + 1)
    n = len(positions)
    scenarios = Scenarios(
        np.zeros((n, problem.horizon), dtype="int64"),
        np.full((n, problem.horizon), problem.lead_times[0], dtype="int64"),
        np.full(n, wday),
        positions,
    )
    rows = {"position": positions}
    for name, policy in policies.items():
        rows[name] = policy(Batch(problem, scenarios))
    return pd.DataFrame(rows)


def evaluate_product(product: dict, demand: pd.DataFrame, artifacts: Path, run, runs: list[dict]) -> dict:
    problem = Problem.from_product(product)
    tuned = load_baselines(artifacts, problem.name)
    real = demand[(demand["product"] == problem.name) & (demand["d"] >= TEST_START)].sort_values("d")
    first_wday = int(real["wday"].iloc[0])
    sets = {
        "test": seeded_scenarios(problem, TEST_SEED, run.test_episodes, first_wday=first_wday),
        "replay": seeded_scenarios(
            problem, REPLAY_SEED, run.replay_episodes, first_wday=first_wday, demand=real["units"].to_numpy()
        ),
    }
    policies = {
        "base_stock": base_stock(tuned["base_stock"]["S"], tuned["base_stock"]["R"]),
        "s_S": s_s(tuned["s_S"]["s"], tuned["s_S"]["S"]),
    }
    run_info = {}
    for spec in runs:
        path = artifacts / "runs" / spec["name"]
        if not (path / "best.zip").exists():
            print(f"  {spec['name']}: not trained, left out")
            continue
        policies[spec["name"]] = model_policy(load_model(path / "best.zip", spec["algorithm"]))
        run_info[spec["name"]] = spec | json.loads((path / "info.json").read_text())

    results, traces = {}, {}
    for set_name, scenarios in sets.items():
        reference = None
        for name in ["s_S", *[p for p in policies if p != "s_S"]]:
            sim = simulate(
                problem, scenarios, policies[name], trace=TRACE_EPISODES if set_name == "test" else 0
            )
            totals = sim["totals"]
            if name == REFERENCE:
                reference = totals["cost"]
            results.setdefault(name, {})[set_name] = summarize(
                totals, None if name == REFERENCE else reference
            )
            if "trace" in sim:
                traces[name] = sim["trace"]
    rows = []
    for name, by_set in results.items():
        info = run_info.get(name, {})
        rows.append(
            {
                "product": problem.name,
                "policy": name,
                "algorithm": info.get("algorithm", name),
                "shaping": info.get("shaping", ""),
                "seed": info.get("seed", np.nan),
                "validation_cost": info.get("best_validation_cost", np.nan),
                **{f"test_{k}": v for k, v in by_set["test"].items()},
                **{f"replay_{k}": v for k, v in by_set["replay"].items()},
            }
        )
    table = pd.DataFrame(rows)
    selected = {}
    for algorithm in ALGORITHMS:
        mine = table[(table["algorithm"] == algorithm) & (table["shaping"] == "cost")]
        if len(mine):
            selected[algorithm] = str(mine.loc[mine["validation_cost"].idxmin(), "policy"])
    best_rl = min(selected.values(), key=lambda p: float(table.set_index("policy").loc[p, "validation_cost"]))
    maps = policy_map(problem, {k: policies[k] for k in ["base_stock", "s_S", *selected.values()]})
    return {"table": table, "selected": selected, "best_rl": best_rl, "traces": traces, "policy_map": maps}


def log_mlflow(artifacts: Path, summary: pd.DataFrame) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(artifacts / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluation"):
        for row in summary.itertuples():
            prefix = f"{row.product}_{row.policy}"
            mlflow.log_metrics(
                {f"{prefix}_test_cost": row.test_cost, f"{prefix}_replay_cost": row.replay_cost}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of the test and replay episodes")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    parser.add_argument("--data-label", default="M5", help="name of the data in the outputs")
    args = parse_args(parser, __file__)
    run = settings(args.sample, args.quick)
    products = load_products(args.data_dir)
    demand = load_demand(args.data_dir)
    plan = planned_runs([p["product"] for p in products], run)
    out = args.artifacts / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    tables, selection = [], {}
    for product in products:
        print(f"{product['product']}:")
        result = evaluate_product(
            product,
            demand,
            args.artifacts,
            run,
            [r for r in plan if r["product"] == product["product"] and r["lead"] == "random"],
        )
        tables.append(result["table"])
        selection[product["product"]] = {"selected": result["selected"], "best_rl": result["best_rl"]}
        np.savez_compressed(
            out / f"trace_{product['product']}.npz",
            **{f"{policy}__{k}": v for policy, trace in result["traces"].items() for k, v in trace.items()},
        )
        result["policy_map"].to_csv(out / f"policy_map_{product['product']}.csv", index=False)
        for row in result["table"].itertuples():
            if row.shaping in ("", "cost") and (
                row.policy in ("base_stock", "s_S") or row.policy in result["selected"].values()
            ):
                print(
                    f"  {row.policy}: test cost {row.test_cost:.1f} per year "
                    f"({row.test_cost_low:.1f} to {row.test_cost_high:.1f}), replay {row.replay_cost:.1f}, "
                    f"fill rate {row.test_fill_rate:.1%}"
                )
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "selection.json").write_text(
        json.dumps(
            {
                "data": args.data_label,
                "episodes": {"test": run.test_episodes, "replay": run.replay_episodes},
                "products": selection,
            },
            indent=2,
        )
    )
    log_mlflow(args.artifacts, summary)


if __name__ == "__main__":
    main()
