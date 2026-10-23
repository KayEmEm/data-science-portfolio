"""Model definitions, preprocessing and cross validated out of fold predictions."""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from loan_default.config import ID_COL, SEED, TARGET_COL, RunSettings

MODEL_NAMES = ["logistic_regression", "lightgbm", "xgboost"]


@dataclass
class Preprocessor:
    """Fix column order and category levels on the training data."""

    columns: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        self.columns = [c for c in frame.columns if c not in (ID_COL, TARGET_COL)]
        self.categories = {
            c: sorted(frame[c].dropna().astype(str).unique().tolist())
            for c in self.columns
            if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reindex(columns=self.columns).copy()
        for col in self.columns:
            if col in self.categories:
                values = out[col].astype("string").astype(object).where(out[col].notna())
                out[col] = pd.Categorical(values, categories=self.categories[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        return out


def _categories_to_str(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object).where(frame.notna(), "missing").astype(str)


def logistic_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    FunctionTransformer(_categories_to_str),
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01),
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("prep", transformer), ("clf", LogisticRegression(max_iter=2000, C=0.1))])


def make_model(name: str, pre: Preprocessor, settings: RunSettings, n_estimators: int | None = None):
    """Return an unfitted estimator. Tree models stop early only when ``n_estimators`` is None."""
    trees = n_estimators or settings.n_estimators
    if name == "logistic_regression":
        categorical = list(pre.categories)
        numeric = [c for c in pre.columns if c not in pre.categories]
        return logistic_pipeline(categorical, numeric)
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=trees,
            learning_rate=settings.learning_rate,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trees,
            learning_rate=settings.learning_rate,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="auc",
            early_stopping_rounds=None if n_estimators else settings.early_stopping_rounds,
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model {name}")


def _fit(name: str, model, X_tr, y_tr, X_va=None, y_va=None, settings: RunSettings | None = None):
    if name == "lightgbm" and X_va is not None:
        model.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(settings.early_stopping_rounds, verbose=False)],
        )
    elif name == "xgboost" and X_va is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model


def _best_iteration(name: str, model) -> int | None:
    if name == "lightgbm":
        return int(model.best_iteration_ or model.n_estimators)
    if name == "xgboost":
        return int(model.best_iteration + 1)
    return None


@dataclass
class FittedModel:
    name: str
    preprocessor: Preprocessor
    estimator: object
    calibrator: IsotonicRegression
    oof_raw: np.ndarray
    n_estimators: int | None

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.preprocessor.transform(frame))[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        return self.calibrator.predict(self.predict_raw(frame))


def fit_with_cv(name: str, train: pd.DataFrame, settings: RunSettings) -> FittedModel:
    """Out of fold predictions, then a final fit on all training rows.

    The isotonic calibrator learns on the out of fold predictions, so it never
    sees a prediction made on a row the model was trained on.
    """
    pre = Preprocessor().fit(train)
    X = pre.transform(train)
    y = train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    iterations = []
    folds = StratifiedKFold(n_splits=settings.n_folds, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in folds.split(X, y):
        model = make_model(name, pre, settings)
        model = _fit(name, model, X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx], settings)
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        if (it := _best_iteration(name, model)) is not None:
            iterations.append(it)
    n_trees = int(np.mean(iterations)) if iterations else None
    final = _fit(name, make_model(name, pre, settings, n_estimators=n_trees), X, y)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
    return FittedModel(name, pre, final, calibrator, oof, n_trees)
