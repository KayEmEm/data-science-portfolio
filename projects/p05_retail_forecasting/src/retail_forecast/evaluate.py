"""Score every stored forecast: WRMSSE on twelve levels, MAPE, reconciliation and intervals.

The model is chosen by mean WRMSSE over the backtest windows. The test window
is then scored once for every model, so the table also shows how far the
backtest ranking carries over.

Prediction intervals come in two kinds:

* native intervals of ETS, Prophet and N-HiTS at level 9 (store x department);
* split conformal intervals for any point forecast, at level 9 and level 12:
  the quantiles of the scaled errors (error divided by the root of the RMSSE
  scale) on the other backtest windows, added to the point forecast. The test
  window uses all backtest windows.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from retail_forecast import reconcile
from retail_forecast.backtest import window_dir
from retail_forecast.config import ARTIFACTS_DIR, DATA_DIR, STORE_DEPT_LEVEL, RunSettings
from retail_forecast.data import load
from retail_forecast.hierarchy import build_hierarchy, store_dept_summing_matrix
from retail_forecast.metrics import Evaluator, coverage, mape, rmsse, scaled_width, scales

EXPERIMENT = "p05_retail_forecasting"
NATIVE_INTERVALS = ["ets", "prophet", "nhits"]
MAPE_LEVELS = (1, 3, 9)
CONFORMAL_LEVELS = (STORE_DEPT_LEVEL, 12)
ENSEMBLE = "ensemble"


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def stored_methods(folder: Path) -> list[str]:
    skip = {"ets_hierarchy", ENSEMBLE}
    return sorted(p.stem for p in folder.glob("*.npz") if p.stem not in skip)


def load_forecasts(artifacts_dir: Path, windows: list[int]) -> dict[int, dict[str, dict]]:
    out = {}
    for origin in windows:
        folder = window_dir(artifacts_dir, origin)
        out[origin] = {m: dict(np.load(folder / f"{m}.npz")) for m in stored_methods(folder)}
    return out


def score_table(evaluators: dict[int, Evaluator], forecasts: dict, levels: dict, kinds: dict) -> pd.DataFrame:
    rows = []
    for origin, by_method in forecasts.items():
        ev = evaluators[origin]
        for method, arrays in by_method.items():
            bottom = arrays["bottom"].astype(float)
            row = {"origin": origin, "window": kinds[origin], "method": method, **ev.score(bottom).to_dict()}
            for n in MAPE_LEVELS:
                row[f"mape_level_{n}"] = mape(ev.actual[n], levels[n].aggregate(bottom))
            rows.append(row)
    return pd.DataFrame(rows)


def reconciliation_table(
    artifacts_dir: Path, evaluators: dict, series: pd.DataFrame, kinds: dict
) -> pd.DataFrame:
    """Weighted RMSSE per level 1 to 9 of ETS base forecasts and of each reconciled set."""
    summing, owners, _ = store_dept_summing_matrix(series)
    owners = np.asarray(owners)
    rows = []
    for origin, ev in evaluators.items():
        folder = window_dir(artifacts_dir, origin)
        if not (folder / "ets_hierarchy.npz").exists():
            continue
        base = np.load(folder / "ets_hierarchy.npz")["base"]
        sets = {"base (incoherent)": base, "bottom_up": reconcile.reconcile(base, summing, "bottom_up")}
        for method in reconcile.METHODS[1:]:
            sets[method] = np.load(folder / f"ets_{method}.npz")["coherent"]
        for name, values in sets.items():
            row = {"origin": origin, "window": kinds[origin], "method": name}
            for n in range(1, STORE_DEPT_LEVEL + 1):
                part = values[owners == n]
                row[f"level_{n}"] = float(np.nansum(ev.weight[n] * rmsse(ev.actual[n], part, ev.scale[n])))
            row["mean_levels_1_9"] = float(np.mean([row[f"level_{n}"] for n in range(1, 10)]))
            rows.append(row)
    return pd.DataFrame(rows)


def scaled_errors(ev: Evaluator, level, bottom: np.ndarray, number: int) -> np.ndarray:
    root = np.sqrt(ev.scale[number])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ev.actual[number] - level.aggregate(bottom)) / root


def conformal_quantiles(
    evaluators: dict,
    forecasts: dict,
    levels: dict,
    method: str,
    number: int,
    c: float,
    calibration: list[int],
) -> tuple[float, float]:
    """Lower and upper quantiles of the scaled errors of ``method`` on the calibration windows."""
    errors = np.concatenate(
        [
            scaled_errors(evaluators[o], levels[number], forecasts[o][method]["bottom"], number).ravel()
            for o in calibration
        ]
    )
    errors = errors[np.isfinite(errors)]
    tail = (1 - c) / 2
    q_low, q_high = np.quantile(errors, [tail, 1 - tail])
    return float(q_low), float(q_high)


def conformal_bounds(point: np.ndarray, scale: np.ndarray, q_low: float, q_high: float):
    root = np.sqrt(np.nan_to_num(scale))[:, None]
    return np.maximum(point + q_low * root, 0), point + q_high * root


def interval_table(
    evaluators: dict, forecasts: dict, levels: dict, kinds: dict, coverages: tuple[float, ...]
) -> pd.DataFrame:
    rows = []
    backtest = [o for o, k in kinds.items() if k == "backtest"]
    for origin, by_method in forecasts.items():
        ev = evaluators[origin]
        for method, arrays in by_method.items():
            for c in coverages:
                tag = int(round(c * 100))
                if method in NATIVE_INTERVALS and f"lower_{tag}" in arrays:
                    lower, upper = arrays[f"lower_{tag}"], arrays[f"upper_{tag}"]
                    rows.append(
                        {
                            "origin": origin,
                            "window": kinds[origin],
                            "method": method,
                            "kind": "native",
                            "level": STORE_DEPT_LEVEL,
                            "nominal": c,
                            "coverage": coverage(ev.actual[STORE_DEPT_LEVEL], lower, upper),
                            "scaled_width": scaled_width(lower, upper, ev.scale[STORE_DEPT_LEVEL]),
                        }
                    )
                calibration = [o for o in backtest if o != origin and method in forecasts[o]]
                if not calibration:
                    continue
                for number in CONFORMAL_LEVELS:
                    q_low, q_high = conformal_quantiles(
                        evaluators, forecasts, levels, method, number, c, calibration
                    )
                    point = levels[number].aggregate(arrays["bottom"].astype(float))
                    lower, upper = conformal_bounds(point, ev.scale[number], q_low, q_high)
                    rows.append(
                        {
                            "origin": origin,
                            "window": kinds[origin],
                            "method": method,
                            "kind": "conformal",
                            "level": number,
                            "nominal": c,
                            "coverage": coverage(ev.actual[number], lower, upper),
                            "scaled_width": scaled_width(lower, upper, ev.scale[number]),
                        }
                    )
    return pd.DataFrame(rows)


def add_ensemble(forecasts: dict, scores: pd.DataFrame) -> str:
    """Mean of LightGBM and the best other model by backtest WRMSSE; returns that partner."""
    backtest = scores[scores["window"] == "backtest"]
    means = backtest.groupby("method")["wrmsse"].mean().drop("lgbm", errors="ignore")
    partner = str(means.idxmin())
    for by_method in forecasts.values():
        if "lgbm" in by_method and partner in by_method:
            bottom = (by_method["lgbm"]["bottom"].astype(float) + by_method[partner]["bottom"]) / 2
            by_method[ENSEMBLE] = {"bottom": bottom}
    return partner


def save_ensemble(artifacts_dir: Path, forecasts: dict) -> None:
    for origin, by_method in forecasts.items():
        if ENSEMBLE in by_method:
            np.savez_compressed(window_dir(artifacts_dir, origin) / f"{ENSEMBLE}.npz", **by_method[ENSEMBLE])


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    backtest = scores[scores["window"] == "backtest"].groupby("method")
    test = scores[scores["window"] == "test"].set_index("method")
    table = pd.DataFrame(
        {
            "backtest_wrmsse": backtest["wrmsse"].mean(),
            "backtest_min": backtest["wrmsse"].min(),
            "backtest_max": backtest["wrmsse"].max(),
            "test_wrmsse": test["wrmsse"],
            **{f"test_mape_level_{n}": test[f"mape_level_{n}"] for n in MAPE_LEVELS},
        }
    )
    return table.sort_values("backtest_wrmsse")


def log_runs(table: pd.DataFrame, scores: pd.DataFrame, settings: RunSettings, data_label: str) -> None:
    for method, row in table.iterrows():
        with mlflow.start_run(run_name=method):
            mlflow.set_tags({"data": data_label, "method": method})
            mlflow.log_params({k: v for k, v in vars(settings).items() if not isinstance(v, tuple)})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if pd.notna(v)})
            for _, window in scores[scores["method"] == method].iterrows():
                mlflow.log_metric("window_wrmsse", float(window["wrmsse"]), step=int(window["origin"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="m5")
    parser.add_argument("--quick", action="store_true")
    args = parse_args(parser, __file__)
    settings = RunSettings.quick() if args.quick else RunSettings()
    plan = json.loads((args.artifacts / "backtest.json").read_text())
    kinds = {o: "backtest" for o in plan["origins"]} | {plan["test_origin"]: "test"}
    tables = load(args.data_dir)
    levels = build_hierarchy(tables["series"])
    evaluators = {o: Evaluator(tables["sales"], tables["prices"], levels, o) for o in kinds}
    forecasts = load_forecasts(args.artifacts, list(kinds))

    scores = score_table(evaluators, forecasts, levels, kinds)
    partner = add_ensemble(forecasts, scores)
    save_ensemble(args.artifacts, forecasts)
    ensemble = {o: {ENSEMBLE: m[ENSEMBLE]} for o, m in forecasts.items() if ENSEMBLE in m}
    scores = pd.concat([scores, score_table(evaluators, ensemble, levels, kinds)], ignore_index=True)
    table = summary(scores)
    selected = str(table.index[0])

    intervals = interval_table(evaluators, forecasts, levels, kinds, settings.interval_levels)
    recon = reconciliation_table(args.artifacts, evaluators, tables["series"], kinds)
    scale_check = {str(o): int(np.isnan(scales(tables["sales"][:, :o])).sum()) for o in kinds}

    out = args.artifacts / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    table.to_csv(out / "summary.csv")
    intervals.to_csv(out / "intervals.csv", index=False)
    recon.to_csv(out / "reconciliation.csv", index=False)
    backtest = [o for o, k in kinds.items() if k == "backtest"]
    conformal = {
        str(number): {
            str(c): conformal_quantiles(evaluators, forecasts, levels, selected, number, c, backtest)
            for c in settings.interval_levels
        }
        for number in CONFORMAL_LEVELS
    }
    meta = {
        "selected": selected,
        "selected_conformal_quantiles": conformal,
        "ensemble_partner": partner,
        "origins": plan["origins"],
        "test_origin": plan["test_origin"],
        "series_without_sales_before_origin": scale_check,
        "data": args.data_label,
    }
    (out / "selection.json").write_text(json.dumps(meta, indent=2))

    setup_mlflow(args.artifacts)
    log_runs(table, scores, settings, args.data_label)
    print(table.round(3).to_string())
    print(f"selected on the backtest windows: {selected}; ensemble partner of lgbm: {partner}")


if __name__ == "__main__":
    main()
