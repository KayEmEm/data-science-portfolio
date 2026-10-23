"""End to end run on synthetic customers: every step reads the previous step's files."""

import json

import pytest

from email_uplift import evaluate, experiment, observational, report, train
from email_uplift.config import Economics, RunSettings


@pytest.fixture(scope="session")
def pipeline(customers_path, tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("artifacts")
    reports = artifacts / "reports"
    experiment.run(customers_path, artifacts, "synthetic")
    training = train.run(
        customers_path,
        artifacts,
        RunSettings.quick(),
        Economics(),
        "synthetic",
        ["t_learner", "causal_forest"],
    )
    evaluation = evaluate.run(artifacts)
    observational.run(customers_path, artifacts, "synthetic", n_replicates=2, n_simulations=3)
    report.run(artifacts, reports)
    return artifacts, reports, training, evaluation


def test_split_sizes_add_up(pipeline, customers):
    _, _, training, _ = pipeline
    assert sum(training["sizes"].values()) == len(customers)
    assert training["sizes"]["test"] == pytest.approx(0.3 * len(customers), abs=1)


def test_selection_uses_validation_only(pipeline):
    _, _, training, evaluation = pipeline
    values = training["validation_policy_values"]
    assert training["selected_policy"] == max(values, key=values.get)
    assert evaluation["selected_policy"] == training["selected_policy"]


def test_evaluation_covers_every_model_and_policy(pipeline):
    _, _, _, evaluation = pipeline
    models = {r["model"] for r in evaluation["qini"]}
    assert models == {"random", "past_spend_rule", "category_rule", "t_learner", "causal_forest"}
    policies = {p["policy"] for p in evaluation["policies"]}
    assert {"no_email", "all_mens", "model_t_learner", "model_causal_forest"} <= policies
    for row in evaluation["policies"]:
        assert row["ci_low"] <= row["value"] <= row["ci_high"]


def test_reports_are_written(pipeline):
    _, reports, _, _ = pipeline
    for name in (
        "experiment_readout.md",
        "ab_results.md",
        "uplift_comparison.md",
        "policy.md",
        "observational.md",
    ):
        assert (reports / name).stat().st_size > 200
    for name in ("effects", "qini_curves", "policy_values", "cost_sensitivity", "observational"):
        assert (reports / "figures" / f"{name}.png").exists()
    memo = (reports / "experiment_readout.md").read_text()
    assert "## Summary" in memo and "Recommendation" in memo
    numbers = json.loads((reports / "results.json").read_text())
    assert numbers["data"] == "synthetic"
