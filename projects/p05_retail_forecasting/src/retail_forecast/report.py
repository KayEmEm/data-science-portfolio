"""Tables, figures and results.json under reports/ from the evaluation outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import apply_style

from retail_forecast.backtest import window_dir
from retail_forecast.config import ARTIFACTS_DIR, DATA_DIR, LEVEL_NAMES, REPORTS_DIR
from retail_forecast.data import load
from retail_forecast.gbm import FEATURES
from retail_forecast.hierarchy import build_hierarchy
from retail_forecast.metrics import scales

# One fixed color per model, from a categorical palette checked for color vision
# deficiency separation; the baseline is drawn in a neutral gray.
COLORS = {
    "lgbm": "#2a78d6",
    "ets": "#eb6834",
    "prophet": "#1baf7a",
    "nhits": "#eda100",
    "ensemble": "#e87ba4",
    "ets_ols": "#008300",
    "ets_wls_struct": "#6250d6",
    "ets_mint_shrink": "#e34948",
    "snaive": "#898781",
}
LABELS = {
    "lgbm": "LightGBM (global, item x store)",
    "ets": "ETS, store x dept, top down",
    "ets_ols": "ETS + OLS reconciliation",
    "ets_wls_struct": "ETS + WLS structural reconciliation",
    "ets_mint_shrink": "ETS + MinT shrink reconciliation",
    "prophet": "Prophet, store x dept, top down",
    "nhits": "N-HiTS (Darts), store x dept, top down",
    "ensemble": "Mean of LightGBM and {partner}",
    "snaive": "Seasonal naive (baseline)",
}
SHORT = {
    "lgbm": "LightGBM",
    "ets": "ETS",
    "prophet": "Prophet",
    "nhits": "N-HiTS",
    "snaive": "seasonal naive",
}
INK = "#0b0b0b"
MUTED = "#52514e"
# Series shown in the forecast figure, one per level of the hierarchy.
EXAMPLES = [(1, "Total"), (3, "CA_3"), (9, "CA_3_FOODS_3"), (12, "FOODS_3_090_CA_3")]


def label(method: str, partner: str) -> str:
    return LABELS.get(method, method).format(partner=SHORT.get(partner, partner))


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    return frame.round(digits).to_markdown()


def wrmsse_figure(summary: pd.DataFrame, partner: str, n_windows: int, path: Path) -> None:
    order = summary.sort_values("backtest_wrmsse", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.2))
    y = np.arange(len(order))
    colors = [COLORS.get(m, MUTED) for m in order.index]
    ax.barh(
        y, order["backtest_wrmsse"], color=colors, height=0.6, label=f"backtest mean ({n_windows} windows)"
    )
    ax.errorbar(
        order["backtest_wrmsse"],
        y,
        xerr=[
            order["backtest_wrmsse"] - order["backtest_min"],
            order["backtest_max"] - order["backtest_wrmsse"],
        ],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    ax.scatter(order["test_wrmsse"], y, marker="D", s=40, color=INK, zorder=3, label="test window")
    for yi, value in zip(y, order["backtest_wrmsse"], strict=True):
        ax.text(
            0.01,
            yi,
            f"{value:.3f}",
            va="center",
            ha="left",
            color=INK,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "#fcfcfb", "ec": "none", "alpha": 0.8},
        )
    ax.set_yticks(y, [label(m, partner) for m in order.index])
    ax.set_xlabel("WRMSSE (lower is better); bars span the backtest minimum to maximum")
    ax.set_title("Accuracy over the twelve M5 levels")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.savefig(path)
    plt.close(fig)


def level_figure(scores: pd.DataFrame, methods: list[str], partner: str, path: Path) -> None:
    backtest = scores[scores["window"] == "backtest"].groupby("method").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 13)
    for method in methods:
        values = backtest.loc[method, [f"level_{n}" for n in x]]
        style = "--" if method == "snaive" else "-"
        ax.plot(
            x,
            values,
            style,
            color=COLORS.get(method, MUTED),
            lw=2,
            marker="o",
            ms=4,
            label=label(method, partner),
        )
    ax.set_xticks(x, [f"{n}\n{LEVEL_NAMES[n].replace(' x ', chr(10) + 'x ')}" for n in x], fontsize=7)
    ax.set_ylabel("weighted RMSSE, backtest mean")
    ax.set_title("Error by level of the hierarchy")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def forecast_figure(tables: dict, artifacts: Path, selection: dict, path: Path) -> None:
    sales, series, calendar = tables["sales"], tables["series"], tables["calendar"]
    levels = build_hierarchy(series)
    origin = selection["test_origin"]
    selected = selection["selected"]
    folder = window_dir(artifacts, origin)
    point_bottom = np.load(folder / f"{selected}.npz")["bottom"].astype(float)
    naive_bottom = np.load(folder / "snaive.npz")["bottom"].astype(float)
    quantiles = selection["selected_conformal_quantiles"]
    dates = pd.DatetimeIndex(calendar["date"])
    past = slice(origin - 56, origin)
    future = slice(origin, origin + 28)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (number, name) in zip(axes.ravel(), EXAMPLES, strict=True):
        level = levels[number]
        if name not in level.names:
            name = level.names[int(np.argmax(level.aggregate(sales[:, :origin]).sum(axis=1)))]
        row = level.names.index(name)
        actual = level.aggregate(sales)[row]
        point = level.aggregate(point_bottom)[row]
        naive = level.aggregate(naive_bottom)[row]
        ax.plot(dates[past], actual[past], color=INK, lw=1.5, label="actual")
        ax.plot(dates[future], actual[future], color=INK, lw=1.5)
        ref = str(number) if str(number) in quantiles else "9"
        scale = scales(level.aggregate(sales[:, :origin])[row : row + 1])[0]
        for c, alpha in (("0.95", 0.18), ("0.8", 0.32)):
            q_low, q_high = quantiles[ref][c]
            lower = np.maximum(point + q_low * np.sqrt(scale), 0)
            upper = point + q_high * np.sqrt(scale)
            ax.fill_between(
                dates[future],
                lower,
                upper,
                color=COLORS.get(selected, MUTED),
                alpha=alpha,
                lw=0,
                label=f"{int(float(c) * 100)}% interval",
            )
        ax.plot(dates[future], point, color=COLORS.get(selected, MUTED), lw=2, label=selected)
        ax.plot(dates[future], naive, "--", color=COLORS["snaive"], lw=1.5, label="seasonal naive")
        ax.axvline(dates[origin - 1], color=MUTED, lw=0.8)
        ax.set_title(f"Level {number} ({LEVEL_NAMES[number]}): {name}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        f"Test window forecasts ({dates[origin].date()} to {dates[origin + 27].date()}), units per day"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def coverage_figure(intervals: pd.DataFrame, partner: str, path: Path) -> None:
    shown = ["snaive", "ets", "prophet", "nhits", "lgbm", "ensemble"]
    test = intervals[(intervals["window"] == "test") & intervals["method"].isin(shown)].copy()
    test["series"] = test["method"] + " (" + test["kind"] + ", level " + test["level"].astype(str) + ")"
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * test["series"].nunique() + 1.5), sharey=True)
    order = sorted(test["series"].unique())
    y = np.arange(len(order))
    for ax, nominal in zip(axes, sorted(test["nominal"].unique()), strict=True):
        part = test[test["nominal"] == nominal].set_index("series").reindex(order)
        colors = [COLORS.get(m, MUTED) for m in part["method"].fillna("")]
        ax.barh(y, part["coverage"], color=colors, height=0.6)
        ax.axvline(nominal, color=INK, lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"nominal {int(nominal * 100)} percent", fontsize=9)
        ax.set_xlabel("share of test days inside the interval")
    axes[0].set_yticks(y, order, fontsize=7)
    fig.suptitle("Prediction interval coverage on the test window")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance_figure(artifacts: Path, origin: int, path: Path) -> pd.Series | None:
    file = window_dir(artifacts, origin) / "lgbm.npz"
    if not file.exists():
        return None
    gain = pd.Series(np.load(file)["importance"], index=FEATURES)
    gain = (gain / gain.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh(gain.index, gain.to_numpy(), color=COLORS["lgbm"], height=0.6)
    ax.set_xlabel("share of total gain")
    ax.set_title("LightGBM feature importance, test window model")
    fig.savefig(path)
    plt.close(fig)
    return gain.sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    apply_style()
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid.axis": "x"})
    evaluation = args.artifacts / "evaluation"
    summary = pd.read_csv(evaluation / "summary.csv", index_col="method")
    scores = pd.read_csv(evaluation / "scores.csv")
    intervals = pd.read_csv(evaluation / "intervals.csv")
    recon = pd.read_csv(evaluation / "reconciliation.csv")
    selection = json.loads((evaluation / "selection.json").read_text())
    partner = selection["ensemble_partner"]
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    tables = load(args.data_dir)
    wrmsse_figure(summary, partner, len(selection["origins"]), figures / "wrmsse_by_model.png")
    shown = [m for m in ["snaive", "ets", "prophet", "nhits", "lgbm", "ensemble"] if m in summary.index]
    level_figure(scores, shown, partner, figures / "wrmsse_by_level.png")
    forecast_figure(tables, args.artifacts, selection, figures / "test_forecasts.png")
    coverage_figure(intervals, partner, figures / "interval_coverage.png")
    gain = importance_figure(args.artifacts, selection["test_origin"], figures / "lgbm_importance.png")

    level_cols = [f"level_{n}" for n in range(1, 13)]
    by_level = scores.groupby(["window", "method"])[level_cols + ["wrmsse"]].mean().loc["backtest"]
    test_by_level = scores[scores["window"] == "test"].set_index("method")[level_cols + ["wrmsse"]]
    recon_cols = [f"level_{n}" for n in range(1, 10)] + ["mean_levels_1_9"]
    recon_backtest = recon[recon["window"] == "backtest"].groupby("method")[recon_cols].mean()
    recon_test = recon[recon["window"] == "test"].set_index("method")[recon_cols]
    interval_summary = (
        intervals.groupby(["window", "method", "kind", "level", "nominal"])[["coverage", "scaled_width"]]
        .mean()
        .unstack("window")
    )
    interval_summary.columns = [f"{a}_{b}" for a, b in interval_summary.columns]

    lines = [
        "# Backtest results",
        "",
        f"Backtest windows: the 28 days after d_{', d_'.join(map(str, selection['origins']))}. "
        f"Test window: the 28 days after d_{selection['test_origin']} (the official M5 evaluation window), "
        "scored once after the model choice.",
        "",
        f"Selected on the backtest windows: **{selection['selected']}**. "
        f"Ensemble: mean of LightGBM and {partner}, the best other model on the backtest windows.",
        "",
        "## WRMSSE and MAPE",
        "",
        "MAPE is computed on the days with positive actual units, at levels 1 (total), 3 (store) and 9 "
        "(store x department).",
        "",
        markdown(summary),
        "",
        "## Weighted RMSSE per level, backtest mean",
        "",
        markdown(by_level.loc[summary.index]),
        "",
        "## Weighted RMSSE per level, test window",
        "",
        markdown(test_by_level.loc[summary.index]),
        "",
        "## Reconciliation of ETS forecasts over levels 1 to 9",
        "",
        "Backtest mean:",
        "",
        markdown(recon_backtest.sort_values("mean_levels_1_9")),
        "",
        "Test window:",
        "",
        markdown(recon_test.loc[recon_backtest.sort_values("mean_levels_1_9").index]),
        "",
        "## Prediction intervals",
        "",
        "Coverage is the share of actual daily values inside the interval; scaled width is the interval "
        "width divided by the root of each series' RMSSE scale.",
        "",
        markdown(interval_summary),
        "",
    ]
    (args.reports / "backtest_results.md").write_text("\n".join(lines))

    test_intervals = intervals[intervals["window"] == "test"]
    results = {
        "selected": selection["selected"],
        "ensemble_partner": partner,
        "summary": summary.round(4).reset_index().to_dict(orient="records"),
        "test_intervals": test_intervals.drop(columns=["origin", "window"])
        .round(4)
        .to_dict(orient="records"),
        "reconciliation_backtest": recon_backtest.round(4).reset_index().to_dict(orient="records"),
        "lgbm_top_features": None if gain is None else gain.head(8).round(4).to_dict(),
        "data": selection.get("data", "m5"),
    }
    (args.reports / "results.json").write_text(json.dumps(results, indent=2))
    timing = {}
    for origin in [*selection["origins"], selection["test_origin"]]:
        file = window_dir(args.artifacts, origin) / "timings.json"
        if file.exists():
            timing[str(origin)] = json.loads(file.read_text())
    (args.reports / "timings.json").write_text(json.dumps(timing, indent=2))
    print(f"report written to {args.reports}")


if __name__ == "__main__":
    main()
