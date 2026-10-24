"""Turn flows into a numeric matrix: signed log of the counts and sizes, standard scaling, and one hot
columns for the most frequent protocol, service and state values. Fitted on the training sample only."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flow_anomaly.config import CATEGORICAL, NUMERIC, TOP_CATEGORIES


def signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass
class FlowPreprocessor:
    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL))
    top: int = TOP_CATEGORIES
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    categories_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, flows: pd.DataFrame) -> "FlowPreprocessor":
        logged = signed_log(flows[self.numeric].to_numpy(dtype="float64"))
        self.mean_ = logged.mean(axis=0)
        scale = logged.std(axis=0)
        self.scale_ = np.where(scale > 1e-9, scale, 1.0)
        self.categories_ = {
            column: flows[column].value_counts().index[: self.top].tolist() for column in self.categorical
        }
        return self

    def transform(self, flows: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit the preprocessor first")
        logged = signed_log(flows[self.numeric].to_numpy(dtype="float64"))
        parts = [np.clip((logged - self.mean_) / self.scale_, -20, 20)]
        for column, values in self.categories_.items():
            codes = pd.Categorical(flows[column], categories=values).codes
            one_hot = np.zeros((len(flows), len(values) + 1))
            one_hot[np.arange(len(flows)), np.where(codes >= 0, codes, len(values))] = 1.0
            parts.append(one_hot)
        return np.hstack(parts).astype("float32")

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for column, values in self.categories_.items():
            names += [f"{column}={v}" for v in values] + [f"{column}=other"]
        return names
