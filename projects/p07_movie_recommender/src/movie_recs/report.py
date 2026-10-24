"""Tables, figures, example lists and results.json under reports/ from the evaluation outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import apply_style

from movie_recs.config import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR, SEED, K
from movie_recs.data import load
from movie_recs.interactions import make_stage
from movie_recs.train import load_lists, stage_dir

# One fixed color per model, from a categorical palette checked for color vision
# deficiency separation; the baselines are drawn in neutral grays.
COLORS = {
    "ranker": "#2a78d6",
    "als": "#eb6834",
    "two_tower": "#1baf7a",
    "popularity_recent": "#898781",
    "popularity": "#b5b3ad",
}
LABELS = {
    "ranker": "ALS + two tower candidates, LightGBM ranker",
    "als": "Implicit ALS",
    "two_tower": "Two tower retrieval (FAISS)",
    "popularity_recent": "Popularity, last 90 days (baseline)",
    "popularity": "Popularity, all time (baseline)",
}
INK = "#0b0b0b"
DEMO_USERS = 3


def markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    return frame.round(digits).to_markdown(index=False)


def metrics_figure(summary: pd.DataFrame, path: Path) -> None:
    test = summary.loc[summary["stage"] == "test"].set_index("model").sort_values(f"ndcg@{K}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 * len(test) + 1.5), sharey=True)
    y = np.arange(len(test))
    colors = [COLORS.get(m, INK) for m in test.index]
    for ax, metric in zip(axes, (f"recall@{K}", f"ndcg@{K}"), strict=True):
        ax.barh(y, test[metric], color=colors, height=0.6)
        for yi, value in zip(y, test[metric], strict=True):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(f"Test {metric.replace('@', ' at ')}")
        ax.set_xlim(0, test[metric].max() * 1.25)
    axes[0].set_yticks(y, [LABELS.get(m, m) for m in test.index])
    fig.suptitle("Top 10 recommendations, test window", x=0.02, ha="left")
    fig.savefig(path)
    plt.close(fig)


def tradeoff_figure(summary: pd.DataFrame, path: Path) -> None:
    test = summary.loc[summary["stage"] == "test"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, (f"coverage@{K}", f"novelty@{K}"), strict=True):
        for model, row in test.iterrows():
            ax.scatter(
                row[metric], row[f"ndcg@{K}"], color=COLORS.get(model, INK), s=60, label=LABELS.get(model)
            )
        ax.set_xlabel(metric.replace("@", " at "))
        ax.set_ylabel(f"NDCG at {K}")
    axes[0].set_title("Catalog coverage", loc="left")
    axes[1].set_title("Novelty (mean bits of self information)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(path)
    plt.close(fig)


def cold_users_figure(cold: pd.DataFrame, path: Path) -> None:
    warm = cold.loc[cold["ratings_before"] != "0 (new users)"].dropna(subset=[f"ndcg@{K}"])
    buckets = list(dict.fromkeys(warm["ratings_before"]))
    models = [m for m in COLORS if m in set(warm["model"])][::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(models)
    for j, model in enumerate(models):
        part = warm.loc[warm["model"] == model].set_index("ratings_before").reindex(buckets)
        ax.bar(
            np.arange(len(buckets)) + j * width,
            part[f"ndcg@{K}"],
            width,
            color=COLORS[model],
            label=LABELS[model],
        )
    counts = warm.drop_duplicates("ratings_before").set_index("ratings_before")["users"].reindex(buckets)
    ax.set_xticks(
        np.arange(len(buckets)) + 0.4 - width / 2, [f"{b}\n({int(n):,} users)" for b, n in counts.items()]
    )
    ax.set_xlabel("Ratings of the user before the test window")
    ax.set_ylabel(f"NDCG at {K}")
    ax.legend(fontsize=8)
    ax.set_title("Test NDCG by user history length", loc="left")
    fig.savefig(path)
    plt.close(fig)


def epochs_figure(epochs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs["epoch"], epochs[f"recall@{K}"], marker="o", color=COLORS["two_tower"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation recall at {K}")
    ax2 = ax.twinx()
    ax2.plot(epochs["epoch"], epochs["loss"], marker="s", color=INK, alpha=0.5)
    ax2.set_ylabel("Training loss")
    ax2.grid(False)
    ax.set_title("Two tower model: choice of the number of epochs", loc="left")
    fig.savefig(path)
    plt.close(fig)


def importance_figure(importance: pd.DataFrame, path: Path) -> None:
    order = importance.sort_values("share")
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(order) + 1))
    ax.barh(order["feature"], order["share"], color=COLORS["ranker"])
    ax.set_xlabel("Share of total gain")
    ax.set_title("Ranking model features", loc="left")
    fig.savefig(path)
    plt.close(fig)


def demo_lists(data: dict, artifacts: Path, model: str, n_users: int) -> str:
    """History and top 10 of the selected model for a few test users, with what they went on to like."""
    stage = make_stage(data["ratings"], data["split"], "test")
    titles = data["movies"].set_index("item")["title"]
    users, items, _ = load_lists(stage_dir(artifacts, "test") / "lists" / f"{model}.npz")
    targets = stage.targets
    liked = targets.groupby("user")["item"].apply(set)
    counts = np.diff(stage.matrix.indptr)[stage.user_rows(users)]
    eligible = np.flatnonzero((counts >= 20) & (counts <= 300))
    rng = np.random.default_rng(SEED)
    picks = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False) if len(eligible) else []
    lines = []
    for row in sorted(picks):
        user = users[row]
        history = stage.history.loc[stage.history["user"] == user].sort_values("ts").tail(8)
        lines.append(f"### User {user} ({counts[row]} ratings before the test window)\n")
        lines.append(
            "Last rated before the window: "
            + "; ".join(
                f"{titles.get(i, i)} ({r:g})" for i, r in zip(history["item"], history["rating"], strict=True)
            )
            + "\n"
        )
        lines.append("| Rank | Recommended | Liked in the window |\n| ---: | --- | :---: |")
        for rank, item in enumerate(items[row][:K], start=1):
            hit = "yes" if item in liked.get(user, set()) else ""
            lines.append(f"| {rank} | {titles.get(item, item)} | {hit} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    apply_style()
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    evaluation = args.artifacts / "evaluation"
    summary = pd.read_csv(evaluation / "summary.csv")
    cold_users = pd.read_csv(evaluation / "cold_users.csv")
    try:
        cold_items = pd.read_csv(evaluation / "cold_items.csv")
    except pd.errors.EmptyDataError:  # no movie in the test window lacked earlier ratings
        cold_items = pd.DataFrame()
    selection = json.loads((evaluation / "selection.json").read_text())
    choices = json.loads((args.artifacts / "selection.json").read_text())
    ranker_info = json.loads((args.artifacts / "ranker.json").read_text())
    importance = pd.read_csv(args.artifacts / "ranker_importance.csv")
    als_grid = pd.read_csv(stage_dir(args.artifacts, "validation") / "als_grid.csv")
    epochs = pd.read_csv(stage_dir(args.artifacts, "validation") / "two_tower_epochs.csv")
    data = load(args.data_dir)
    timings = {
        s: json.loads((stage_dir(args.artifacts, s) / "timings.json").read_text())
        for s in ("validation", "test")
    }

    metrics_figure(summary, figures / "metrics_by_model.png")
    tradeoff_figure(summary, figures / "coverage_novelty.png")
    cold_users_figure(cold_users, figures / "cold_start_users.png")
    epochs_figure(epochs, figures / "two_tower_epochs.png")
    importance_figure(importance, figures / "ranker_importance.png")

    split = data["split"]
    ratings = data["ratings"]
    windows = []
    for name in ("validation", "test"):
        stage = make_stage(ratings, split, name)
        windows.append(
            {
                "stage": name,
                "window_start": str(pd.Timestamp(stage.start, unit="s")),
                "window_end": str(pd.Timestamp(stage.end, unit="s")),
                "history_ratings": len(stage.history),
                "catalog_movies": len(stage.items),
                "window_ratings": len(stage.window),
                "liked_ratings": len(stage.targets),
                "users_scored": int(len(stage.eval_users())),
                "users_without_history": int(len(stage.new_users())),
                "liked_ratings_on_new_movies": float((~stage.targets["item"].isin(stage.items)).mean()),
            }
        )
    windows = pd.DataFrame(windows)

    def table(stage: str) -> pd.DataFrame:
        part = summary.loc[summary["stage"] == stage].drop(columns="stage").copy()
        part["model"] = part["model"].map(lambda m: LABELS.get(m, m))
        return part.sort_values(f"ndcg@{K}", ascending=False)

    sections = [
        "# Project 7 results\n",
        f"Data: {len(ratings):,} ratings. "
        f"Selected on validation NDCG at {K}: **{LABELS.get(selection['selected'])}**.\n",
        "## Windows\n",
        markdown(windows),
        f"\n## Test window (all scored users)\n\n{markdown(table('test'))}",
        f"\n## Validation window (comparison half of the users)\n\n{markdown(table('validation'))}",
        f"\n## ALS settings (validation)\n\n{markdown(als_grid)}",
        f"\n## Two tower epochs (validation sample)\n\n{markdown(epochs)}",
        f"\n## Cold start: users by history length (test)\n\n{markdown(cold_users)}",
        "\n## Cold start: movies first rated in the test window\n",
        markdown(cold_items) if len(cold_items) else "none",
        f"\n## Ranking model features\n\n{markdown(importance)}",
        f"\n## Run times (seconds)\n\n{markdown(pd.DataFrame(timings).reset_index(names='step'), 1)}\n",
    ]
    (args.reports / "results.md").write_text("\n".join(sections))
    (args.reports / "demo_lists.md").write_text(
        f"# Example recommendations (test window)\n\nModel: {LABELS[selection['selected']]}.\n\n"
        + demo_lists(data, args.artifacts, selection["selected"], DEMO_USERS)
    )
    results = {
        "data": selection["data"],
        "selected": selection["selected"],
        "choices": choices,
        "ranker": ranker_info,
        "windows": windows.to_dict(orient="records"),
        "summary": summary.round(4).to_dict(orient="records"),
        "cold_users": cold_users.round(4).to_dict(orient="records"),
        "cold_items": cold_items.round(4).to_dict(orient="records") if len(cold_items) else [],
    }
    (args.reports / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"reports written to {args.reports}")


if __name__ == "__main__":
    main()
