"""Heterogeneous effect models: who gains most from each email.

Every model predicts, per customer, the incremental two week spend of the
mens email and of the womens email against no email. The randomization
fixes the propensity, so learners that need one get the design shares
(``DummyClassifier``) instead of a fitted model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.metalearners import TLearner, XLearner
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier

from email_uplift.config import ARMS, SEED, RunSettings
from email_uplift.data import TREATMENT_COL, feature_matrix

ARM_CODES = [1, 2]  # treatment codes of ARMS, see data.TREATMENT_CODES
BASELINES = ["random", "past_spend_rule", "category_rule"]
MODELS = ["t_learner", "x_learner", "causal_forest", "dr_learner_causalml"]


def causalml_available() -> bool:
    try:
        import causalml.inference.meta  # noqa: F401
    except ImportError:
        return False
    return True


def _regressor(settings: RunSettings) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=settings.n_estimators,
        learning_rate=settings.learning_rate,
        num_leaves=settings.num_leaves,
        min_child_samples=settings.min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )


@dataclass
class UpliftModel:
    """A fitted model that returns one effect column per email arm."""

    name: str
    estimator: object

    def effects(self, x: pd.DataFrame) -> np.ndarray:
        """Predicted effect of each arm against control, shape (n, 2)."""
        if self.name in ("t_learner", "x_learner", "causal_forest"):
            return np.column_stack(
                [self.estimator.effect(x.to_numpy(), T0=0, T1=code).ravel() for code in ARM_CODES]
            )
        if self.name == "dr_learner_causalml":
            return np.asarray(self.estimator.predict(x.to_numpy(), verbose=False))
        raise ValueError(self.name)


def fit_model(name: str, frame: pd.DataFrame, outcome: str, settings: RunSettings) -> UpliftModel:
    x = feature_matrix(frame).to_numpy()
    y = frame[outcome].to_numpy(dtype=float)
    t = frame[TREATMENT_COL].to_numpy()
    if name == "t_learner":
        est = TLearner(models=_regressor(settings))
        est.fit(y, t, X=x)
    elif name == "x_learner":
        est = XLearner(
            models=_regressor(settings),
            propensity_model=DummyClassifier(strategy="prior"),
        )
        est.fit(y, t, X=x)
    elif name == "causal_forest":
        est = CausalForestDML(
            model_y=_regressor(settings),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=settings.forest_trees,
            min_samples_leaf=settings.forest_min_leaf,
            max_samples=0.4,
            cv=3,
            random_state=SEED,
        )
        est.fit(y, t, X=x)
    elif name == "dr_learner_causalml":
        from causalml.inference.meta import BaseDRRegressor

        groups = np.asarray(["control", "mens", "womens"])[t]
        # Design propensity of each arm within the pair (arm, control), as CausalML expects.
        p = {}
        for arm in ("mens", "womens"):
            pair = np.isin(groups, ["control", arm])
            p[arm] = np.full(len(groups), (groups[pair] == arm).mean())
        est = BaseDRRegressor(learner=_regressor(settings), control_name="control")
        est.fit(x, treatment=groups, y=y, p=p, seed=SEED)
    else:
        raise ValueError(f"unknown model {name}")
    return UpliftModel(name, est)


def baseline_scores(name: str, frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Targeting scores of the simple rules, shape (n, 2). Higher means send first.

    ``random`` ranks customers in random order. ``past_spend_rule`` sends
    to the best past customers first, the usual habit without a model.
    ``category_rule`` sends each email to buyers of its category first,
    then by past spend.
    """
    n = len(frame)
    if name == "random":
        return np.random.default_rng(seed).random((n, len(ARMS)))
    history = frame["history"].to_numpy(dtype=float)
    if name == "past_spend_rule":
        return np.column_stack([history, history])
    if name == "category_rule":
        scale = history / (history.max() + 1)
        return np.column_stack([frame["mens"] + scale, frame["womens"] + scale]).astype(float)
    raise ValueError(f"unknown baseline {name}")
