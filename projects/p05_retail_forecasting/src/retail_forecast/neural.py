"""One global neural model from Darts (N-HiTS) on the store x department series.

Optional: it needs the ``neural`` extra (Darts with PyTorch). ``available()``
tells the pipeline whether to run it.
"""

import importlib.util
import logging
import os
import warnings

import numpy as np

from retail_forecast.config import HORIZON, SEED
from retail_forecast.statistical import Intervals

QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]


def available() -> bool:
    return importlib.util.find_spec("darts") is not None and importlib.util.find_spec("torch") is not None


def nhits_forecast(
    history: np.ndarray,
    horizon: int = HORIZON,
    epochs: int = 30,
    coverages: tuple[float, ...] = (0.8, 0.95),
    samples: int = 200,
    seed: int = SEED,
) -> tuple[np.ndarray, Intervals]:
    """Fit one N-HiTS model on all series (log1p units, scaled per series) and forecast."""
    os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_WARNINGS", "1")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.ERROR)
    import torch
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import NHiTSModel
    from darts.utils.likelihood_models import QuantileRegression

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    series = [TimeSeries.from_values(np.log1p(y.astype("float32"))) for y in history]
    scaler = Scaler()
    scaled = scaler.fit_transform(series)
    input_length = min(112, history.shape[1] - horizon - 1)
    model = NHiTSModel(
        input_chunk_length=input_length,
        output_chunk_length=horizon,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=256,
        dropout=0.1,
        batch_size=64,
        n_epochs=epochs,
        likelihood=QuantileRegression(QUANTILES),
        random_state=seed,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "enable_checkpointing": False,
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaled, max_samples_per_ts=300, verbose=False)
        predictions = model.predict(n=horizon, series=scaled, num_samples=samples, verbose=False)
    # draws: series x horizon x samples (Darts keeps components on axis 1; one component here)
    draws = np.stack([np.expm1(p.all_values()[:, 0, :]) for p in scaler.inverse_transform(predictions)])
    point = np.maximum(np.median(draws, axis=2), 0)
    bounds = {}
    for c in coverages:
        tail = (1 - c) / 2
        lower = np.maximum(np.quantile(draws, tail, axis=2), 0)
        upper = np.maximum(np.quantile(draws, 1 - tail, axis=2), 0)
        bounds[c] = (lower, upper)
    return point, bounds
