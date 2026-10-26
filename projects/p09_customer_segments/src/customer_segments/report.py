"""Tables, figures, the interactive segment map and results.json under reports/ from the evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from customer_segments.categories import CATEGORIES
from customer_segments.config import ARTIFACTS_DIR, DATA_DIR, MIN_SEGMENT_SHARE, MIN_STABILITY, REPORTS_DIR
from customer_segments.features import SHARES

# Categorical slots in fixed order; the selected item in blue, the others in gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SELECTED = "#2a78d6"
MUTED = "#c3c2b7"
BASELINE = "#898781"
INK = "#0b0b0b"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "triangle-down", "star"]
NAMES = {
    "kmeans_rfm_mix": "k means, RFM and product mix (selected)",
    "kmeans_first_attempt": "k means, first attempt with return share",
    "kmeans_rfm": "k means, RFM only",
    "hdbscan": "HDBSCAN, RFM and product mix",
    "rfm_grid": "RFM grid, ten rule segments (baseline)",
}
PROFILE_COLUMNS = [
    "segment",
    "customers",
    "customer_share",
    "spend_share",
    "median_recency_days",
    "median_frequency",
    "median_monetary",
    "median_mean_basket",
    "mean_return_share",
    "top_categories",
    "future_purchase_rate",
    "future_spend_mean",
    "future_spend_share",
    "jaccard",
]
PROFILE_FEATURES = {
    "median_recency_days": "Recency (days)",
    "median_frequency": "Orders",
    "median_monetary": "Net spend (GBP)",
    "median_mean_basket": "Basket (GBP)",
    "median_tenure_days": "Tenure (days)",
    "median_distinct_products": "Products",
    "mean_return_share": "Return share",
    "future_purchase_rate": "Bought next quarter",
    "future_spend_mean": "Spend next quarter (GBP)",
}


def selection_figure(
    grid: pd.DataFrame, stab: pd.DataFrame, gap: pd.DataFrame, sel: dict, path: Path
) -> None:
    table = grid.merge(stab, on=["feature_set", "scaling", "k"]).merge(
        gap[["feature_set", "scaling", "k", "gap", "gap_se"]], on=["feature_set", "scaling", "k"]
    )
    table = table.loc[table["feature_set"] == "rfm_mix"]
    scalings = sorted(table["scaling"].unique(), key=["raw", "standard", "robust", "quantile"].index)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = (
        ("silhouette", "Silhouette (higher is better)"),
        ("gap", "Gap statistic"),
        ("stability_ari", "Bootstrap stability (adjusted Rand index)"),
    )
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for i, sc in enumerate(scalings):
            part = table.loc[table["scaling"] == sc].sort_values("k")
            width = 2.5 if sc == sel["scaling"] else 1.5
            ax.plot(
                part["k"], part[metric], marker="o", markersize=4, linewidth=width, color=SERIES[i], label=sc
            )
        if metric == "stability_ari":
            ax.axhline(MIN_STABILITY, color=BASELINE, linewidth=1, linestyle="--")
        ax.set_xlabel("k")
        ax.set_title(title, loc="left", fontsize=10)
    best = table.loc[(table["scaling"] == sel["scaling"]) & (table["k"] == sel["k"])].iloc[0]
    axes[0].scatter(
        [sel["k"]], [best["silhouette"]], s=120, facecolors="none", edgecolors=INK, linewidths=1.5
    )
    axes[2].legend(title="Scaling", fontsize=8)
    fig.suptitle(
        f"Choice of k on RFM and product mix features: {sel['scaling']} scaling, k = {sel['k']} (circled)",
        x=0.02,
        y=1.03,
        ha="left",
    )
    fig.savefig(path)
    plt.close(fig)


def umap_figure(segments: pd.DataFrame, path: Path) -> None:
    """Small multiples: each panel shows one segment in blue over all customers in gray."""
    ids = sorted(segments["segment"].unique())
    cols = min(4, len(ids))
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, seg in zip(axes.ravel(), ids, strict=False):
        part = segments["segment"] == seg
        ax.scatter(segments["umap_x"], segments["umap_y"], s=2, color=MUTED, linewidths=0)
        ax.scatter(
            segments.loc[part, "umap_x"], segments.loc[part, "umap_y"], s=3, color=SELECTED, linewidths=0
        )
        ax.set_title(f"Segment {seg} ({part.mean():.0%})", fontsize=9, loc="left")
    fig.suptitle("Customers on a UMAP map of the clustering features", x=0.02, y=1.02, ha="left")
    fig.savefig(path)
    plt.close(fig)


def profile_figure(profiles: pd.DataFrame, path: Path) -> None:
    table = profiles.set_index("segment")[list(PROFILE_FEATURES)]
    # Each column as a z score across segments, so that colors compare segments within a feature.
    z = (table - table.mean()) / table.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(table.columns) + 2.5, 0.5 * len(table) + 1.8))
    image = ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        text = (
            f"{v:.0%}" if table.columns[j] in ("mean_return_share", "future_purchase_rate") else f"{v:,.0f}"
        )
        ax.text(j, i, text, ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(
        range(len(table.columns)), [PROFILE_FEATURES[c] for c in table.columns], rotation=30, ha="right"
    )
    ax.set_yticks(range(len(table)), [f"Segment {s}" for s in table.index])
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="Below (blue) or above (red) the segment average")
    ax.set_title("Segment profiles: medians before the cutoff, outcome after it", loc="left")
    fig.savefig(path)
    plt.close(fig)


def mix_figure(profiles: pd.DataFrame, path: Path) -> None:
    table = profiles.set_index("segment")[SHARES]
    table.columns = [c.removeprefix("share_").replace("_", " ") for c in table.columns]
    fig, ax = plt.subplots(figsize=(0.8 * len(table.columns) + 2.5, 0.5 * len(table) + 1.8))
    image = ax.imshow(
        table.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.3, float(table.max().max())), aspect="auto"
    )
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.25 else INK)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table)), [f"Segment {s}" for s in table.index])
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="Share of purchase value")
    ax.set_title("Product mix per segment (mean share of purchase value before the cutoff)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def outcome_figure(comparison: pd.DataFrame, path: Path) -> None:
    order = comparison.iloc[::-1]
    y = np.arange(len(order))
    colors = [
        SELECTED if n == "kmeans_rfm_mix" else BASELINE if n == "rfm_grid" else MUTED
        for n in order["segmentation"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(order) + 1.8), sharey=True)
    panels = (("spend", "Log spend next quarter"), ("purchase", "Bought next quarter (yes or no)"))
    for ax, (target, title) in zip(axes, panels, strict=True):
        value = order[f"eta2_{target}"].to_numpy()
        low, high = order[f"eta2_{target}_low"].to_numpy(), order[f"eta2_{target}_high"].to_numpy()
        ax.barh(y, value, color=colors, height=0.6)
        ax.errorbar(
            value, y, xerr=[value - low, high - value], fmt="none", ecolor=INK, elinewidth=1, capsize=2
        )
        for yi, v, h in zip(y, value, high, strict=True):
            ax.text(h + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(0.1, float(comparison[f"eta2_{target}_high"].max())) * 1.25)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Variance explained by segment (adjusted eta squared)")
    axes[0].set_yticks(y, [NAMES.get(n, n) for n in order["segmentation"]])
    fig.suptitle("How well each segmentation separates next quarter's behavior", x=0.02, y=1.03, ha="left")
    fig.savefig(path)
    plt.close(fig)


def interactive_map(table: pd.DataFrame, segments: pd.DataFrame, path: Path) -> None:
    """Plotly map of all customers, without customer ids; hover shows the rounded profile values."""
    frame = table.join(segments[["segment", "umap_x", "umap_y", "rfm_grid"]])
    top = frame[SHARES].idxmax(axis=1).str.removeprefix("share_").str.replace("_", " ")
    fig = go.Figure()
    for i, seg in enumerate(sorted(frame["segment"].unique())):
        part = frame["segment"] == seg
        hover = [
            f"Segment {seg}<br>RFM grid: {g.replace('_', ' ')}<br>Last order {r:.0f} days before the cutoff"
            f"<br>{f:.0f} orders, {m:,.0f} GBP net<br>Largest category: {c}"
            for g, r, f, m, c in zip(
                frame.loc[part, "rfm_grid"],
                frame.loc[part, "recency_days"],
                frame.loc[part, "frequency"],
                frame.loc[part, "monetary"],
                top[part],
                strict=True,
            )
        ]
        fig.add_trace(
            go.Scattergl(
                x=frame.loc[part, "umap_x"].round(3),
                y=frame.loc[part, "umap_y"].round(3),
                mode="markers",
                name=f"Segment {seg} ({part.sum():,})",
                marker={"size": 6, "color": SERIES[i % len(SERIES)], "symbol": SYMBOLS[i % len(SYMBOLS)]},
                text=hover,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Customer segments on a UMAP map (click a legend entry to hide a segment, double click to"
        " isolate it)",
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={"itemsizing": "constant"},
        height=700,
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def table_md(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def results_markdown(
    audit: dict,
    info: dict,
    sel: dict,
    comparison: pd.DataFrame,
    profiles: pd.DataFrame,
    ablation: pd.DataFrame,
    density: pd.DataFrame,
    grid: pd.DataFrame,
    stab: pd.DataFrame,
    data_label: str,
) -> str:
    comp = comparison.assign(segmentation=comparison["segmentation"].map(NAMES))
    comp = comp[
        [
            "segmentation",
            "segments",
            "eta2_spend",
            "eta2_spend_low",
            "eta2_spend_high",
            "eta2_purchase",
            "eta2_purchase_low",
            "eta2_purchase_high",
            "purchase_rate_min",
            "purchase_rate_max",
        ]
    ]
    prof = profiles[PROFILE_COLUMNS]
    prof_ablation = ablation[PROFILE_COLUMNS]
    kgrid = grid.merge(stab, on=["feature_set", "scaling", "k"])
    kgrid = kgrid.loc[
        kgrid["feature_set"] != "rfm",
        [
            "feature_set",
            "scaling",
            "k",
            "silhouette",
            "davies_bouldin",
            "min_share",
            "stability_ari",
            "stability_ari_p05",
        ],
    ]
    dens = density[["scaling", "method", "segments", "noise_share", "silhouette"]]
    candidates = ", ".join(f"{c['scaling']} k={c['k']}" for c in sel["candidates"]) or "none"
    lines = [
        f"# Project 9 results ({data_label})",
        "",
        "## Data",
        "",
        f"* {audit['rows_read']:,} rows read from `{audit['source_file']}`"
        f" (documented {audit['documented_rows']:,}), {audit['duplicate_rows']:,} exact duplicates dropped"
        f" ({audit.get('rows_in_two_sheets', 0):,} of them"
        f" in both yearly sheets), {audit['rows_without_customer']:,} rows without a customer id dropped,"
        f" {audit['rows_kept']:,} kept.",
        f"* {audit['customers']:,} customers, {audit['invoices']:,} invoices, {audit['first_date'][:10]} to"
        f" {audit['last_date'][:10]}; {audit['return_lines']:,} return lines; spend in no category rule:"
        f" {audit['other_category_spend_share']:.1%}.",
        f"* Cutoff {info['cutoff']}: features from the orders before it, outcome from the"
        f" {info['outcome_days']} days after it. {info['customers']:,} customers with positive net spend"
        " before the cutoff;"
        f" {info['future_purchase_share']:.1%} of them bought again after it.",
        "",
        "## Selection",
        "",
        f"* Selected: k means on RFM and product mix features, {sel['scaling']} scaling, k = {sel['k']}"
        f" ({sel['rule']}): silhouette {sel['silhouette']:.3f},"
        f" bootstrap stability {sel['stability_ari']:.3f},"
        f" smallest segment {sel['min_share']:.1%}.",
        f"* Candidates (stability at least {MIN_STABILITY},"
        f" smallest segment at least {MIN_SEGMENT_SHARE:.0%}):"
        f" {candidates}.",
        f"* Mean stability over the k range by scaling: {sel['mean_stability_by_scaling']}.",
        f"* Gap statistic choice by scaling (None: no k in the range meets the rule): {sel['gap_choice']}.",
        f"* RFM only: {sel['rfm_only']['scaling']} scaling, k = {sel['rfm_only']['k']}.",
        f"* First attempt with return share: {sel['first_attempt']['scaling']} scaling,"
        f" k = {sel['first_attempt']['k']}"
        f" ({sel['first_attempt']['rule']}): silhouette {sel['first_attempt']['silhouette']:.3f},"
        f" bootstrap stability {sel['first_attempt']['stability_ari']:.3f},"
        f" smallest segment {sel['first_attempt']['min_share']:.1%}.",
        f"* HDBSCAN setting compared: `{sel['hdbscan']}`.",
        "",
        "## Next quarter check",
        "",
        table_md(comp),
        "",
        "Adjusted eta squared: share of the variance of the outcome explained by the segment means, adjusted"
        " for the number of segments; 95 percent bootstrap intervals over customers.",
        "",
        "## Segment profiles",
        "",
        table_md(prof, ".2f"),
        "",
        "Segments are numbered by their share of net spend before the cutoff. `jaccard`: mean overlap of the"
        " segment with its best match over bootstrap refits (above 0.75 is usually read as stable).",
        "",
        "## Segment profiles of the first attempt with return share",
        "",
        table_md(prof_ablation, ".2f"),
        "",
        "## k means grid",
        "",
        table_md(kgrid),
        "",
        "## Density based clustering (RFM and product mix)",
        "",
        table_md(dens),
        "",
    ]
    return "\n".join(lines)


def run(data_dir: Path, artifacts: Path, reports: Path, data_label: str) -> None:
    apply_style()
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ev = artifacts / "evaluation"
    audit = json.loads((data_dir / "audit.json").read_text())
    info = json.loads((artifacts / "customers.json").read_text())
    sel = json.loads((ev / "selection.json").read_text())
    table = pd.read_parquet(artifacts / "customers.parquet")
    segments = pd.read_parquet(ev / "segments.parquet")
    comparison = pd.read_csv(ev / "comparison.csv")
    profiles = pd.read_csv(ev / "profiles.csv")
    ablation = pd.read_csv(ev / "profiles_first_attempt.csv")
    grid = pd.read_csv(artifacts / "kmeans_grid.csv")
    stab = pd.read_csv(artifacts / "stability.csv")
    gap = pd.read_csv(artifacts / "gap.csv")
    density = pd.read_csv(artifacts / "density.csv")

    selection_figure(grid, stab, gap, sel, figures / "k_selection.png")
    umap_figure(segments, figures / "umap_segments.png")
    profile_figure(profiles, figures / "segment_profiles.png")
    mix_figure(profiles, figures / "product_mix.png")
    outcome_figure(comparison, figures / "next_quarter.png")
    interactive_map(table, segments, reports / "segment_map.html")

    (reports / "results.md").write_text(
        results_markdown(audit, info, sel, comparison, profiles, ablation, density, grid, stab, data_label)
    )
    results = {
        "data": data_label,
        "cutoff": info["cutoff"],
        "customers": info["customers"],
        "selection": sel,
        "comparison": comparison.round(4).to_dict("records"),
        "profiles": profiles.round(4).to_dict("records"),
        "profiles_first_attempt": ablation.round(4).to_dict("records"),
        "categories": CATEGORIES,
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"report written to {reports}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--data-label", default="Online Retail II")
    args = parse_args(parser, __file__)
    run(args.data_dir, args.artifacts, args.reports, args.data_label)


if __name__ == "__main__":
    main()
