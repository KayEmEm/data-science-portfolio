import numpy as np
import pytest

from email_uplift.config import RunSettings
from email_uplift.data import feature_matrix
from email_uplift.synthetic import true_visit_effect
from email_uplift.uplift import MODELS, baseline_scores, causalml_available, fit_model

FAST = RunSettings.quick()


@pytest.mark.parametrize("name", MODELS)
def test_models_find_who_responds_to_each_email(customers, name):
    if name == "dr_learner_causalml" and not causalml_available():
        pytest.skip("causalml not installed")
    model = fit_model(name, customers, "visit", FAST)
    effects = model.effects(feature_matrix(customers))
    assert effects.shape == (len(customers), 2)
    for j, arm in enumerate(("Mens E-Mail", "Womens E-Mail")):
        truth = true_visit_effect(customers, arm)
        assert np.corrcoef(effects[:, j], truth)[0, 1] > 0.2


def test_baseline_scores_have_one_column_per_arm(customers):
    for name in ("random", "past_spend_rule", "category_rule"):
        scores = baseline_scores(name, customers)
        assert scores.shape == (len(customers), 2)
    rule = baseline_scores("category_rule", customers)
    mens = customers["mens"].to_numpy() == 1
    assert rule[mens, 0].min() > rule[~mens, 0].max()


def test_unknown_model_is_rejected(customers):
    with pytest.raises(ValueError):
        fit_model("s_learner", customers, "visit", FAST)
