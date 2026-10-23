"""Forecast every window with every model and store the forecasts under artifacts/.

Windows are the rolling backtest origins and the test origin of ``Backtest``.
Each model sees only the days up to the origin. Bottom level (item x store)
forecasts are stored for every model, so that every model is scored the same
way on all twelve levels:

* ``snaive`` and ``lgbm`` forecast the item x store series directly;
* ``ets``, ``prophet`` and ``nhits`` forecast the 70 store x department series,
  split down by each item's share of the last 28 days (top down);
* ``ets_<method>`` reconcile ETS forecasts of all 154 series of levels 1 to 9
  before the split.

Finished windows are skipped, so an interrupted run resumes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast import gbm, neural, reconcile, statistical
from retail_forecast.config import ARTIFACTS_DIR, DATA_DIR, STORE_DEPT_LEVEL, Backtest, RunSettings
from retail_forecast.data import load
from retail_forecast.hierarchy import bottom_shares, build_hierarchy, disaggregate, store_dept_summing_matrix

MODELS = ["snaive", "ets", "prophet", "lgbm", "nhits"]


def closures(sales: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Days on which total sales fall below ``threshold`` of the median day (store closures)."""
    total = sales.sum(axis=0)
    return total < threshold * np.median(total)


def window_dir(artifacts_dir: Path, origin: int) -> Path:
    return artifacts_dir / "windows" / f"origin_{origin}"


def save(path: Path, bottom: np.ndarray, intervals: dict | None = None, **extra: np.ndarray) -> None:
    arrays = {"bottom": bottom.astype("float32"), **extra}
    for coverage, (lower, upper) in (intervals or {}).items():
        tag = int(round(coverage * 100))
        arrays[f"lower_{tag}"] = lower.astype("float32")
        arrays[f"upper_{tag}"] = upper.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_window(tables: dict, origin: int, models: list[str], settings: RunSettings, out: Path, force: bool):
    sales, prices, series, calendar = tables["sales"], tables["prices"], tables["series"], tables["calendar"]
    levels = build_hierarchy(series)
    level9 = levels[STORE_DEPT_LEVEL]
    summing, owners, _ = store_dept_summing_matrix(series)
    observed = sales[:, :origin]
    start = max(0, origin - settings.history_days)
    aggregates = np.vstack([levels[n].aggregate(observed[:, start:]) for n in range(1, STORE_DEPT_LEVEL + 1)])
    # The stores close on Christmas Day; the aggregate models see a filled value instead.
    aggregates = statistical.fill_closures(aggregates, closures(sales)[start:origin])
    store_dept = aggregates[-summing.shape[1] :]
    shares = bottom_shares(observed, level9)
    dates = pd.DatetimeIndex(calendar["date"].iloc[start:origin])
    coverages = settings.interval_levels
    timings = {}

    def done(name: str) -> bool:
        return (out / f"{name}.npz").exists() and not force

    for model in models:
        if done(model):
            continue
        began = time.perf_counter()
        if model == "snaive":
            save(out / "snaive.npz", statistical.seasonal_naive(observed))
        elif model == "ets":
            base, bounds, residuals = statistical.ets_forecast(aggregates, coverages=coverages)
            np.savez_compressed(out / "ets_hierarchy.npz", base=base, owners=np.asarray(owners))
            level9_bounds = {
                c: (lo[-len(store_dept) :], hi[-len(store_dept) :]) for c, (lo, hi) in bounds.items()
            }
            for method in reconcile.METHODS[1:]:
                coherent = reconcile.reconcile(base, summing, method, residuals=residuals)
                bottom = disaggregate(np.maximum(coherent[-len(store_dept) :], 0), level9, shares)
                save(out / f"ets_{method}.npz", bottom, coherent=coherent)
            save(out / "ets.npz", disaggregate(base[-len(store_dept) :], level9, shares), level9_bounds)
        elif model == "prophet":
            point, bounds = statistical.prophet_forecast(store_dept, dates, calendar, coverages=coverages)
            save(out / "prophet.npz", disaggregate(point, level9, shares), bounds)
        elif model == "lgbm":
            point, booster = gbm.fit_predict(
                sales,
                prices,
                series,
                calendar,
                origin,
                train_days=settings.gbm_train_days,
                rounds=settings.gbm_rounds,
                leaves=settings.gbm_leaves,
                learning_rate=settings.gbm_learning_rate,
            )
            importance = booster.feature_importance(importance_type="gain")
            save(out / "lgbm.npz", point, importance=importance)
        elif model == "nhits":
            if not neural.available():
                print("  nhits skipped: Darts with PyTorch is not installed (pip install -e '.[neural]')")
                continue
            point, bounds = neural.nhits_forecast(
                store_dept, epochs=settings.neural_epochs, coverages=coverages
            )
            save(out / "nhits.npz", disaggregate(point, level9, shares), bounds)
        else:
            raise ValueError(f"unknown model {model}")
        timings[model] = round(time.perf_counter() - began, 1)
        print(f"  {model}: {timings[model]} s", flush=True)
    timing_path = out / "timings.json"
    previous = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing_path.write_text(json.dumps(previous | timings, indent=2))


def origins(backtest: Backtest, which: str) -> list[int]:
    if which == "backtest":
        return list(backtest.origins)
    if which == "test":
        return [backtest.test_origin]
    return [*backtest.origins, backtest.test_origin]


def backtest_for(n_days: int) -> Backtest:
    """The M5 windows, or for a shorter synthetic table, three windows that end at its last day."""
    full = Backtest()
    if full.test_origin + full.horizon == n_days:
        return full
    h = full.horizon
    last = n_days - h
    return Backtest(origins=(last - 2 * h, last - h), test_origin=last, horizon=h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--windows", choices=["backtest", "test", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="small models for smoke runs")
    parser.add_argument("--force", action="store_true", help="recompute finished windows")
    args = parser.parse_args()
    settings = RunSettings.quick() if args.quick else RunSettings()
    tables = load(args.data_dir)
    backtest = backtest_for(tables["sales"].shape[1])
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "backtest.json").write_text(
        json.dumps({"origins": list(backtest.origins), "test_origin": backtest.test_origin}, indent=2)
    )
    for origin in origins(backtest, args.windows):
        print(f"window after d_{origin}", flush=True)
        out = window_dir(args.artifacts, origin)
        out.mkdir(parents=True, exist_ok=True)
        run_window(tables, origin, args.models, settings, out, args.force)


if __name__ == "__main__":
    main()
