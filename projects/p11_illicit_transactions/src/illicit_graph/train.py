"""Fit every model and store its illicit probability for every transaction.

Models, all trained on the labelled transactions of the training steps:

* ``logistic_local``: logistic regression on the 93 local features (baseline);
* ``lightgbm_local``, ``lightgbm_all``: LightGBM on the local features, and on the local and the
  aggregated ones (the feature only reference of the plan);
* ``lightgbm_all_graph``: LightGBM on all features and the structural measures of ``network.py``;
* ``gcn_all``, ``sage_all``, ``sage_local``: GCN and GraphSAGE on all features, and GraphSAGE on
  the local features only, each with three seeds; their stored probability is the mean over seeds.

LightGBM stops on validation log loss. Every finished model is stored under
``artifacts/models/<name>/`` and skipped when the command runs again, so a stopped run resumes.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from illicit_graph import gnn
from illicit_graph.config import (
    AGGREGATED,
    ARTIFACTS_DIR,
    DATA_DIR,
    GNN_SEEDS,
    ILLICIT,
    LOCAL,
    QUICK_GNN,
    QUICK_LIGHTGBM,
    SEED,
    UNKNOWN,
    GNNSettings,
    LightGBMSettings,
)
from illicit_graph.data import load
from illicit_graph.network import STRUCTURAL, load_metrics

MODELS = {
    "logistic_local": {
        "kind": "logistic",
        "features": "local",
        "label": "Logistic regression, local (baseline)",
    },
    "lightgbm_local": {"kind": "lightgbm", "features": "local", "label": "LightGBM, local"},
    "lightgbm_all": {"kind": "lightgbm", "features": "all", "label": "LightGBM, all features"},
    "lightgbm_all_graph": {
        "kind": "lightgbm",
        "features": "all_graph",
        "label": "LightGBM, all features and graph measures",
    },
    "gcn_all": {"kind": "gcn", "features": "all", "label": "GCN, all features"},
    "sage_all": {"kind": "sage", "features": "all", "label": "GraphSAGE, all features"},
    "sage_local": {"kind": "sage", "features": "local", "label": "GraphSAGE, local"},
}
BASELINE = "logistic_local"
GNN_KINDS = {"gcn", "sage"}


def feature_matrix(
    nodes: pd.DataFrame, metrics: pd.DataFrame | None, name: str
) -> tuple[np.ndarray, list[str]]:
    columns = LOCAL if name == "local" else LOCAL + AGGREGATED
    x = nodes[columns].to_numpy(dtype="float32")
    if name == "all_graph":
        graph = metrics[STRUCTURAL].to_numpy(dtype="float32")
        graph[:, [0, 1, 5, 6]] = np.log1p(graph[:, [0, 1, 5, 6]])  # counts on a log scale
        x = np.hstack([x, graph])
        columns = columns + STRUCTURAL
    return x, list(columns)


def masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    split = nodes["split"].astype(str).to_numpy()
    labelled = nodes["label"].to_numpy() != UNKNOWN
    return {name: labelled & (split == name) for name in ("train", "validation", "test")} | {
        "train_steps": split == "train",
        "validation_steps": split == "validation",
    }


def fit_tabular(kind: str, x: np.ndarray, y: np.ndarray, m: dict, settings: LightGBMSettings):
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        model.fit(x[m["train"]], y[m["train"]])
        return model, {}
    model = lgb.LGBMClassifier(
        n_estimators=settings.n_estimators,
        learning_rate=settings.learning_rate,
        num_leaves=settings.num_leaves,
        min_child_samples=settings.min_child_samples,
        subsample=settings.subsample,
        subsample_freq=1,
        colsample_bytree=settings.colsample_bytree,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(
        x[m["train"]],
        y[m["train"]],
        eval_set=[(x[m["validation"]], y[m["validation"]])],
        callbacks=[lgb.early_stopping(settings.early_stopping, verbose=False)],
    )
    return model, {"best_iteration": int(model.best_iteration_ or settings.n_estimators)}


def run_model(
    name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame | None,
    artifacts: Path,
    lightgbm: LightGBMSettings,
    gnn_settings: GNNSettings,
    seeds: list[int],
    force: bool,
) -> None:
    spec = MODELS[name]
    out = artifacts / "models" / name
    if (out / "scores.npy").exists() and not force:
        print(f"{name}: stored")
        return
    out.mkdir(parents=True, exist_ok=True)
    x, columns = feature_matrix(nodes, metrics, spec["features"])
    y = (nodes["label"].to_numpy() == ILLICIT).astype("int64")
    m = masks(nodes)
    start = time.perf_counter()
    info = {"model": name, **spec, "features_count": len(columns)}
    if spec["kind"] in GNN_KINDS:
        edge_array = edges[["src", "dst"]].to_numpy()
        per_seed = []
        for seed in seeds:
            seed_path = out / f"scores_seed{seed}.npy"
            if seed_path.exists() and not force:
                per_seed.append(np.load(seed_path))
                continue
            prob, history = gnn.fit_predict(
                spec["kind"],
                x,
                edge_array,
                y,
                m["train"],
                m["validation"],
                m["train_steps"],
                m["validation_steps"],
                gnn_settings,
                seed,
            )
            np.save(seed_path, prob)
            pd.DataFrame(history).to_csv(out / f"history_seed{seed}.csv", index=False)
            per_seed.append(prob)
            best = max(history, key=lambda row: row["validation_pr_auc"])
            print(
                f"  {name} seed {seed}: best epoch {best['epoch']} of {len(history)}, "
                f"validation PR AUC {best['validation_pr_auc']:.3f}"
            )
        scores = np.mean(per_seed, axis=0)
        info["seeds"] = list(seeds)
    else:
        model, extra = fit_tabular(spec["kind"], x, y, m, lightgbm)
        scores = model.predict_proba(x)[:, 1].astype("float64")
        joblib.dump({"model": model, "columns": columns}, out / "model.joblib")
        info.update(extra)
    info["seconds"] = round(time.perf_counter() - start, 1)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    np.save(out / "scores.npy", scores)
    print(f"{name}: {len(columns)} features, {info['seconds']} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--only", nargs="*", help="models to fit (default: all)")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    parser.add_argument("--force", action="store_true", help="refit stored models")
    args = parse_args(parser, __file__)
    nodes, edges = load(args.data_dir)
    metrics = load_metrics(args.artifacts)
    m = masks(nodes)
    y = nodes["label"].to_numpy() == ILLICIT
    for part in ("train", "validation", "test"):
        print(f"{part}: {m[part].sum():,} labelled transactions, {y[m[part]].mean():.1%} illicit")
    for name in MODELS:
        if args.only and name not in args.only:
            continue
        run_model(
            name,
            nodes,
            edges,
            metrics,
            args.artifacts,
            QUICK_LIGHTGBM if args.quick else LightGBMSettings(),
            QUICK_GNN if args.quick else GNNSettings(),
            GNN_SEEDS[:2] if args.quick else GNN_SEEDS,
            args.force,
        )


if __name__ == "__main__":
    main()
