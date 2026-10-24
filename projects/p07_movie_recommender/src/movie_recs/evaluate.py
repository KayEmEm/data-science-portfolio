"""Offline evaluation of every model on the validation comparison users and on the test users.

Outputs under ``artifacts/evaluation/``:

* ``summary.csv``: recall, NDCG, hit rate, coverage and novelty at 10 per stage and model;
* ``cold_users.csv``: test recall and NDCG at 10 by the number of ratings a user had before the window,
  and for users with no rating before it (popularity only);
* ``cold_items.csv``: movies first rated in the test window, ranked among themselves;
* ``paired_test.csv``: test NDCG difference of each model to two references, paired by user.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from movie_recs import baselines
from movie_recs.config import ARTIFACTS_DIR, DATA_DIR, RECENT_DAYS, K
from movie_recs.data import load
from movie_recs.interactions import make_stage
from movie_recs.metrics import half_width, per_user, summarize
from movie_recs.train import load_lists, stage_dir

MODELS = ["popularity", "popularity_recent", "als", "two_tower", "ranker"]
EXPERIMENT = "p07_movie_recommender"
BUCKETS = [(1, 9), (10, 49), (50, 199), (200, None)]


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def rows_for(users: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    rows = np.searchsorted(users, wanted)
    if (rows >= len(users)).any() or (users[np.minimum(rows, len(users) - 1)] != wanted).any():
        raise ValueError("a list lacks some of the users to score")
    return rows


def bucket_name(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo} to {hi}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--data-label", default="MovieLens 32M")
    args = parser.parse_args()
    data = load(args.data_dir)
    out = args.artifacts / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    rows, cold_users, cold_items, test_ndcg = [], [], [], {}
    for name in ("validation", "test"):
        stage = make_stage(data["ratings"], data["split"], name)
        lists = stage_dir(args.artifacts, name) / "lists"
        compare = load_lists(lists / "ranker.npz")[0]  # validation: half B; test: every user
        popularity = stage.history["item"].value_counts()
        n_history_users = len(stage.users)
        history_counts = pd.Series(np.diff(stage.matrix.indptr), index=stage.users)
        for model in MODELS:
            users, items, _ = load_lists(lists / f"{model}.npz")
            items = items[rows_for(users, compare)]
            metrics = summarize(
                compare, items, stage.targets, popularity, n_history_users, len(stage.items), K
            )
            rows.append({"stage": name, "model": model, **metrics})
            if name != "test":
                continue
            scores = per_user(compare, items, stage.targets, K)
            test_ndcg[model] = scores["ndcg"].to_numpy()
            scores["n_history"] = history_counts.reindex(compare).to_numpy()
            for lo, hi in BUCKETS:
                upper = np.inf if hi is None else hi
                part = scores.loc[scores["n_history"].between(lo, upper)]
                cold_users.append(
                    {
                        "model": model,
                        "ratings_before": bucket_name(lo, hi),
                        "users": len(part),
                        f"recall@{K}": part["recall"].mean(),
                        f"ndcg@{K}": part["ndcg"].mean(),
                    }
                )
        if name == "test":
            new_users = stage.new_users()
            scores = baselines.popularity_scores(stage, RECENT_DAYS)
            top = stage.items[np.argsort(-scores, kind="stable")[:K]]
            new_scores = per_user(new_users, np.tile(top, (len(new_users), 1)), stage.targets, K)
            cold_users.append(
                {
                    "model": "popularity_recent",
                    "ratings_before": "0 (new users)",
                    "users": len(new_users),
                    f"recall@{K}": new_scores["recall"].mean(),
                    f"ndcg@{K}": new_scores["ndcg"].mean(),
                }
            )
            targets = stage.targets
            new_item = ~targets["item"].isin(stage.items)
            for model in ("two_tower_new_items", "random_new_items"):
                path = lists / f"{model}.npz"
                if not path.exists():
                    continue
                users, items, _ = load_lists(path)
                metrics = per_user(users, items, targets.loc[new_item], K)
                cold_items.append(
                    {
                        "model": model,
                        "users": len(users),
                        "new_movies": int(targets.loc[new_item, "item"].nunique()),
                        "share_of_liked_ratings_on_new_movies": float(new_item.mean()),
                        f"recall@{K}": metrics["recall"].mean(),
                        f"ndcg@{K}": metrics["ndcg"].mean(),
                        f"hit_rate@{K}": (metrics["hits"] > 0).mean(),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    paired = []
    for model in MODELS:
        for reference in ("popularity_recent", "als"):
            if model == reference:
                continue
            diff = pd.Series(test_ndcg[model] - test_ndcg[reference])
            paired.append(
                {
                    "model": model,
                    "reference": reference,
                    f"ndcg@{K}_difference": diff.mean(),
                    "ci95": half_width(diff),
                }
            )
    pd.DataFrame(paired).to_csv(out / "paired_test.csv", index=False)
    pd.DataFrame(cold_users).to_csv(out / "cold_users.csv", index=False)
    pd.DataFrame(cold_items).to_csv(out / "cold_items.csv", index=False)
    validation = summary.loc[summary["stage"] == "validation"].set_index("model")
    selected = validation[f"ndcg@{K}"].idxmax()
    (out / "selection.json").write_text(
        json.dumps({"selected": selected, "by": f"validation ndcg@{K}", "data": args.data_label}, indent=2)
    )

    setup_mlflow(args.artifacts)
    for row in rows:
        with mlflow.start_run(run_name=f"{row['stage']}_{row['model']}"):
            mlflow.set_tags({"data": args.data_label, "stage": row["stage"], "model": row["model"]})
            mlflow.log_metrics(
                {k.replace("@", "_at_"): float(v) for k, v in row.items() if k not in ("stage", "model")}
            )
    print(summary.round(4).to_string(index=False))
    print(f"selected on validation: {selected}")
    if (summary["users"] == 0).any() or (summary[f"recall@{K}"].isna()).any():
        raise ValueError("a model was scored on no users")


if __name__ == "__main__":
    main()
