"""Fit the uplift models on visits and on spend, and choose one on a validation split.

The customers are split into fit (52.5 percent), validation (17.5 percent)
and test (30 percent), stratified by arm. Models learn on the fit split. The
model whose targeting policy earns the most on the validation split is
selected. The test split is scored once, for reporting, by ``evaluate``.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from portfolio_common.seed import set_seed
from sklearn.model_selection import train_test_split

from email_uplift.config import ARTIFACTS_DIR, CUSTOMERS_PATH, SEED, Economics, RunSettings
from email_uplift.data import TREATMENT_COL, feature_matrix, load
from email_uplift.targeting import choose_arms, policy_value, qini_coefficient, rewards
from email_uplift.uplift import ARM_CODES, BASELINES, MODELS, baseline_scores, causalml_available, fit_model

EXPERIMENT = "p04_email_uplift"
OUTCOMES = ["visit", "spend"]  # visit carries more signal; spend is what the policy is paid on
POLICY_OUTCOME = "spend"
ARM_NAMES = ("mens", "womens")
KEEP_COLUMNS = [TREATMENT_COL, "visit", "conversion", "spend", "history", "recency", "mens", "womens"]


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{artifacts_dir / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def split(frame: pd.DataFrame, settings: RunSettings) -> dict[str, pd.DataFrame]:
    train, test = train_test_split(
        frame, test_size=settings.test_size, stratify=frame[TREATMENT_COL], random_state=SEED
    )
    fit, validation = train_test_split(
        train, test_size=settings.validation_size, stratify=train[TREATMENT_COL], random_state=SEED
    )
    return {"fit": fit, "validation": validation, "test": test}


def qini_by_arm(scores: np.ndarray, part: pd.DataFrame, outcome: str) -> list[float]:
    """Qini coefficient of each arm, on the rows of that arm and of control."""
    t = part[TREATMENT_COL].to_numpy()
    out = []
    for j, code in enumerate(ARM_CODES):
        pair = np.isin(t, [0, code])
        out.append(qini_coefficient(scores[pair, j], part[outcome].to_numpy()[pair], t[pair] == code))
    return out


def candidate_policies(part: pd.DataFrame, models: list[str], economics: Economics) -> dict[str, np.ndarray]:
    """Arm per customer under each policy: three uniform ones, a rule and one per spend model."""
    n = len(part)
    policies = {
        "no_email": np.zeros(n, dtype=int),
        "all_mens": np.full(n, 1),
        "all_womens": np.full(n, 2),
        # mens email to every buyer of mens merchandise, womens email to the rest
        "category_rule": np.where(part["mens"].to_numpy() == 1, 1, 2),
    }
    for name in models:
        columns = [f"score_{POLICY_OUTCOME}_{name}_{arm}" for arm in ARM_NAMES]
        policies[f"model_{name}"] = choose_arms(part[columns].to_numpy(), economics)
    return policies


def available_models(requested: list[str]) -> list[str]:
    if "dr_learner_causalml" in requested and not causalml_available():
        print("causalml is not installed: skipping dr_learner_causalml (pip install -e '.[causalml]')")
        return [m for m in requested if m != "dr_learner_causalml"]
    return requested


def run(
    customers_path: Path,
    artifacts_dir: Path,
    settings: RunSettings,
    economics: Economics,
    data_label: str,
    models: list[str],
) -> dict:
    set_seed(SEED)
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(artifacts_dir)
    parts = split(load(customers_path), settings)
    models = available_models(models)

    outputs = {name: parts[name][KEEP_COLUMNS].reset_index(drop=True) for name in ("validation", "test")}
    summary = {
        "data": data_label,
        "outcomes": OUTCOMES,
        "policy_outcome": POLICY_OUTCOME,
        "sizes": {name: len(part) for name, part in parts.items()},
        "economics": asdict(economics),
        "settings": asdict(settings),
        "validation": {outcome: {} for outcome in OUTCOMES},
    }
    val = parts["validation"]
    val_t = val[TREATMENT_COL].to_numpy()
    val_reward = rewards(val[POLICY_OUTCOME], val_t, economics)
    for name in BASELINES:
        for split_name in ("validation", "test"):
            scores = baseline_scores(name, parts[split_name])
            for j, arm in enumerate(ARM_NAMES):
                outputs[split_name][f"score_{name}_{arm}"] = scores[:, j]
        val_scores = outputs["validation"][[f"score_{name}_{arm}" for arm in ARM_NAMES]].to_numpy()
        for outcome in OUTCOMES:
            summary["validation"][outcome][name] = {"qini": qini_by_arm(val_scores, val, outcome)}
    for outcome in OUTCOMES:
        for name in models:
            with mlflow.start_run(run_name=f"{name}_{outcome}"):
                model = fit_model(name, parts["fit"], outcome, settings)
                columns = [f"score_{outcome}_{name}_{arm}" for arm in ARM_NAMES]
                for split_name in ("validation", "test"):
                    effects = model.effects(feature_matrix(parts[split_name]))
                    outputs[split_name][columns] = effects
                val_effects = outputs["validation"][columns].to_numpy()
                result = {"qini": qini_by_arm(val_effects, val, outcome)}
                metrics = {f"validation_qini_{a}": q for a, q in zip(ARM_NAMES, result["qini"], strict=True)}
                if outcome == POLICY_OUTCOME:
                    policy = choose_arms(val_effects, economics)
                    result["policy_value"] = policy_value(policy, val_t, val_reward)
                    metrics["validation_policy_value"] = result["policy_value"]
                summary["validation"][outcome][name] = result
                mlflow.set_tags({"data": data_label, "outcome": outcome})
                params = {"model": name, "outcome": outcome, **asdict(settings), **asdict(economics)}
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

    # Selection uses the validation split only: the policy with the highest expected
    # profit per customer, among the model policies and the simple ones.
    summary["models"] = models
    policies = candidate_policies(outputs["validation"], models, economics)
    values = {name: policy_value(arms, val_t, val_reward) for name, arms in policies.items()}
    summary["validation_policy_values"] = values
    summary["selected_model"] = max(models, key=lambda m: values[f"model_{m}"])
    summary["selected_policy"] = max(values, key=values.get)
    for split_name, frame in outputs.items():
        frame.to_parquet(artifacts_dir / f"{split_name}_scores.parquet", index=False)
    (artifacts_dir / "training.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=Path, default=CUSTOMERS_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="small settings for a smoke run")
    parser.add_argument("--data-label", default="hillstrom-2008", help="recorded with every output")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--margin", type=float, default=Economics.margin)
    parser.add_argument("--cost-per-email", type=float, default=Economics.cost_per_email)
    args = parser.parse_args()
    settings = RunSettings.quick() if args.quick else RunSettings()
    economics = Economics(margin=args.margin, cost_per_email=args.cost_per_email)
    summary = run(args.customers, args.artifacts, settings, economics, args.data_label, args.models)
    print(json.dumps(summary["validation"], indent=1))
    print(f"selected model: {summary['selected_model']}; selected policy: {summary['selected_policy']}")


if __name__ == "__main__":
    main()
