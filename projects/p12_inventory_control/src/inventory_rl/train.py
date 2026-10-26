"""Train PPO and SAC (Stable Baselines3) on each product, with several seeds, and the shaping study.

Runs:

* main: every product x {PPO, SAC} x the seeds of the run settings, trained on the cost reward;
* shaping study: PPO on ``config.SHAPING_PRODUCT`` with the ``potential`` and ``service``
  rewards and the shaping seeds; its ``cost`` runs are the main runs;
* fixed lead time: PPO on the slow product with the lead time of ``dp.py``, scored there
  against the exact optimum.

Every ``eval_every`` steps the current policy is scored by its mean yearly cost on the
validation scenarios (a seed stream of its own); the checkpoint with the lowest cost is kept as
``best.zip``. That curve is the learning curve of the report. A finished run is skipped when the
command runs again, so a stopped run resumes; runs go to ``artifacts/runs/<run>/``. MLflow logs
each finished run under ``artifacts/mlflow.db``.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from inventory_rl.config import (
    ALGORITHMS,
    ARTIFACTS_DIR,
    DATA_DIR,
    GAMMA,
    SHAPING_PRODUCT,
    VALIDATION_SEED,
    PPOSettings,
    RunSettings,
    SACSettings,
    settings,
)
from inventory_rl.data import load_products
from inventory_rl.dp import DP_LEAD, DP_PRODUCT

EXPERIMENT = "p12_inventory_control"


def run_name(product: str, algorithm: str, shaping: str, seed: int, lead: str = "random") -> str:
    middle = "fixedlead" if lead == "fixed" else shaping
    return f"{product}_{algorithm}_{middle}_s{seed}"


def planned_runs(products: list[str], run: RunSettings) -> list[dict]:
    plan = [
        {"product": p, "algorithm": a, "shaping": "cost", "seed": s}
        for p in products
        for a in ALGORITHMS
        for s in run.seeds
    ]
    if SHAPING_PRODUCT in products:
        plan += [
            {"product": SHAPING_PRODUCT, "algorithm": "ppo", "shaping": shaping, "seed": s}
            for shaping in ("potential", "service")
            for s in run.shaping_seeds
        ]
    if DP_PRODUCT in products:
        # PPO on the setting of the dynamic program (dp.py), to measure its gap to the exact optimum.
        plan += [
            {"product": DP_PRODUCT, "algorithm": "ppo", "shaping": "cost", "seed": s, "lead": "fixed"}
            for s in run.shaping_seeds
        ]
    plan = [{"lead": "random"} | r for r in plan]
    return [
        r | {"name": run_name(r["product"], r["algorithm"], r["shaping"], r["seed"], r["lead"])} for r in plan
    ]


def build_model(algorithm: str, env, seed: int, steps: int):
    from stable_baselines3 import PPO, SAC

    if algorithm == "ppo":
        p = PPOSettings()
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=p.learning_rate,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=GAMMA,
            gae_lambda=p.gae_lambda,
            clip_range=p.clip_range,
            ent_coef=p.ent_coef,
            policy_kwargs={"net_arch": list(p.net)},
            seed=seed,
            device="cpu",
            verbose=0,
        )
    s = SACSettings()
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=s.learning_rate,
        buffer_size=min(s.buffer_size, steps),
        batch_size=s.batch_size,
        learning_starts=min(s.learning_starts, steps // 4),
        tau=s.tau,
        gamma=GAMMA,
        policy_kwargs={"net_arch": list(s.net)},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def train_one(spec: dict, product: dict, base_level: int, run: RunSettings, out: Path) -> dict:
    """One training run; returns its info. Runs in a worker process with one torch thread."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    from inventory_rl.dynamics import Problem, model_policy, seeded_scenarios, simulate
    from inventory_rl.env import InventoryEnv

    torch.set_num_threads(1)
    problem = Problem.from_product(product)
    if spec["lead"] == "fixed":
        problem = replace(problem, lead_times=(DP_LEAD,), lead_probs=(1.0,))
    validation = seeded_scenarios(problem, VALIDATION_SEED, run.validation_episodes)
    steps = run.ppo_steps if spec["algorithm"] == "ppo" else run.sac_steps
    n_envs = run.n_envs if spec["algorithm"] == "ppo" else 1
    env = make_vec_env(
        InventoryEnv,
        n_envs=n_envs,
        seed=spec["seed"],
        env_kwargs={"problem": problem, "shaping": spec["shaping"], "base_level": base_level},
    )
    model = build_model(spec["algorithm"], env, spec["seed"], steps)
    out.mkdir(parents=True, exist_ok=True)
    curve: list[dict] = []
    start = time.perf_counter()

    def score() -> float:
        return float(simulate(problem, validation, model_policy(model))["totals"]["cost"].mean())

    class Validate(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_check = run.eval_every
            self.best = np.inf

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_check:
                self.next_check += run.eval_every
                self.record()
            return True

        def record(self) -> None:
            cost = score()
            curve.append(
                {"step": self.num_timesteps, "cost": cost, "seconds": round(time.perf_counter() - start, 1)}
            )
            if cost < self.best:
                self.best = cost
                model.save(out / "best.zip")

    callback = Validate()
    model.learn(total_timesteps=steps, callback=callback)
    if not curve or curve[-1]["step"] < model.num_timesteps:
        callback.num_timesteps = model.num_timesteps
        callback.record()
    model.save(out / "final.zip")
    pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
    best = min(curve, key=lambda row: row["cost"])
    info = spec | {
        "steps": int(model.num_timesteps),
        "n_envs": n_envs,
        "best_step": int(best["step"]),
        "best_validation_cost": best["cost"],
        "final_validation_cost": curve[-1]["cost"],
        "seconds": round(time.perf_counter() - start, 1),
        "settings": asdict(PPOSettings() if spec["algorithm"] == "ppo" else SACSettings()),
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return info


def log_mlflow(artifacts: Path, info: dict, curve: pd.DataFrame) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(artifacts / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=info["name"]):
        mlflow.log_params(
            {k: info[k] for k in ("product", "algorithm", "shaping", "seed", "steps", "n_envs")}
        )
        mlflow.log_params({f"hp_{k}": v for k, v in info["settings"].items()})
        mlflow.log_metrics(
            {k: info[k] for k in ("best_validation_cost", "final_validation_cost", "best_step", "seconds")}
        )
        for row in curve.itertuples():
            mlflow.log_metric("validation_cost", row.cost, step=int(row.step))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of steps, seeds and episodes")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="runs in parallel")
    parser.add_argument("--only", nargs="*", help="run names to train (default: all)")
    parser.add_argument("--force", action="store_true", help="train finished runs again")
    args = parse_args(parser, __file__)
    from inventory_rl.baselines import load_baselines

    run = settings(args.sample, args.quick)
    products = {p["product"]: p for p in load_products(args.data_dir)}
    plan = [r for r in planned_runs(list(products), run) if not args.only or r["name"] in args.only]
    todo = []
    for spec in plan:
        out = args.artifacts / "runs" / spec["name"]
        if (out / "info.json").exists() and not args.force:
            print(f"{spec['name']}: stored")
            continue
        tuned = load_baselines(args.artifacts, spec["product"])["s_S"]
        base_level = (tuned["s"] + tuned["S"]) / 2  # reference position of the potential shaping
        todo.append((spec, products[spec["product"]], base_level, run, out))
    todo.sort(key=lambda job: job[0]["algorithm"] != "sac")  # the longest runs first
    print(f"{len(todo)} of {len(plan)} runs to train, {args.jobs} at a time")
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(train_one, *job): job[0]["name"] for job in todo}
        for future in as_completed(futures):
            info = future.result()
            curve = pd.read_csv(args.artifacts / "runs" / info["name"] / "curve.csv")
            log_mlflow(args.artifacts, info, curve)
            print(
                f"{info['name']}: best validation cost {info['best_validation_cost']:.1f} at step "
                f"{info['best_step']:,} of {info['steps']:,}, {info['seconds']:.0f} s"
            )


if __name__ == "__main__":
    main()
