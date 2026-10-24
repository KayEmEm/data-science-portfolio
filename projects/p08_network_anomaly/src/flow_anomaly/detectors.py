"""The detectors under one interface: ``fit`` on a matrix, ``score`` returns higher values for flows
that look more unusual. All but the supervised reference see no labels."""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from flow_anomaly import autoencoder
from flow_anomaly.config import (
    HBOS_BINS,
    LOF_NEIGHBORS,
    SEED,
    AutoencoderSettings,
    ForestSettings,
    SupervisedSettings,
)


@dataclass(frozen=True)
class Spec:
    label: str
    sample: str  # "main" (FIT_ROWS), "neighbor" (NEIGHBOR_ROWS) or "labeled" (supervised)


SPECS = {
    "robust_z": Spec("Robust z-score (baseline)", "main"),
    "iforest": Spec("Isolation Forest", "main"),
    "lof": Spec("Local Outlier Factor", "neighbor"),
    "hbos": Spec("HBOS (PyOD)", "main"),
    "ecod": Spec("ECOD (PyOD)", "main"),
    "copod": Spec("COPOD (PyOD)", "main"),
    "autoencoder": Spec("Autoencoder", "main"),
    "ensemble": Spec("Ensemble: Isolation Forest, ECOD, autoencoder", "main"),
    "lightgbm": Spec("LightGBM, supervised (upper reference)", "labeled"),
}
UNSUPERVISED = [name for name, spec in SPECS.items() if spec.sample != "labeled"]
ENSEMBLE_MEMBERS = ["iforest", "ecod", "autoencoder"]


class RobustZ:
    """Mean absolute distance from the median in units of the interquartile range, capped at 10."""

    def fit(self, x: np.ndarray) -> "RobustZ":
        self.median_ = np.median(x, axis=0)
        iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
        self.scale_ = np.where(iqr > 1e-6, iqr, x.std(axis=0) + 1e-6)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.abs((x - self.median_) / self.scale_), 10).mean(axis=1)


class Forest:
    def __init__(self, settings: ForestSettings, seed: int = SEED):
        self.model = IsolationForest(
            n_estimators=settings.n_estimators, max_samples=settings.max_samples, random_state=seed, n_jobs=-1
        )

    def fit(self, x: np.ndarray) -> "Forest":
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(x)


class LOF:
    def __init__(self, neighbors: int = LOF_NEIGHBORS):
        self.model = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, n_jobs=-1)

    def fit(self, x: np.ndarray) -> "LOF":
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray, batch: int = 50_000) -> np.ndarray:
        return np.concatenate([-self.model.score_samples(x[i : i + batch]) for i in range(0, len(x), batch)])


class PyOD:
    def __init__(self, model):
        self.model = model

    def fit(self, x: np.ndarray) -> "PyOD":
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(x), dtype="float64")


class AutoencoderDetector:
    def __init__(self, settings: AutoencoderSettings, folder: Path, seed: int = SEED):
        self.settings = settings
        self.folder = folder
        self.seed = seed

    def fit(self, x: np.ndarray) -> "AutoencoderDetector":
        self.history = autoencoder.fit(x, self.settings, self.folder, self.seed)
        self.model = autoencoder.load(self.folder, x.shape[1], self.settings)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return autoencoder.feature_errors(self.model, x).mean(axis=1).astype("float64")


class Ensemble:
    """Mean of the members' scores, each turned into its quantile among the training sample scores."""

    def __init__(self, members: dict):
        self.members = members

    def fit(self, x: np.ndarray) -> "Ensemble":
        self.reference_ = {name: np.sort(m.score(x)) for name, m in self.members.items()}
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        quantiles = [
            np.searchsorted(self.reference_[name], m.score(x), side="right") / len(self.reference_[name])
            for name, m in self.members.items()
        ]
        return np.mean(quantiles, axis=0)


class Supervised:
    def __init__(self, settings: SupervisedSettings, seed: int = SEED):
        self.settings = settings
        self.model = lgb.LGBMClassifier(
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            num_leaves=settings.num_leaves,
            min_child_samples=settings.min_child_samples,
            subsample=settings.subsample,
            subsample_freq=1,
            colsample_bytree=settings.colsample_bytree,
            random_state=seed,
            verbose=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "Supervised":
        self.model.fit(
            x,
            y,
            eval_X=x_val,
            eval_y=y_val,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(self.settings.early_stopping, verbose=False)],
        )
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


def make(
    name: str,
    forest: ForestSettings,
    ae: AutoencoderSettings,
    folder: Path,
    members: dict | None = None,
    seed: int = SEED,
):
    if name == "robust_z":
        return RobustZ()
    if name == "iforest":
        return Forest(forest, seed)
    if name == "lof":
        return LOF()
    if name == "hbos":
        return PyOD(HBOS(n_bins=HBOS_BINS))
    if name == "ecod":
        return PyOD(ECOD())
    if name == "copod":
        return PyOD(COPOD())
    if name == "autoencoder":
        return AutoencoderDetector(ae, folder / "autoencoder", seed)
    if name == "ensemble":
        return Ensemble({m: members[m] for m in ENSEMBLE_MEMBERS})
    raise KeyError(name)
