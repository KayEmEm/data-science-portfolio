"""Select the segmentation, check every segmentation on the next quarter, and profile the segments.

Selection uses no outcome. A k means solution on the ``rfm_mix`` features is a candidate when its
mean bootstrap adjusted Rand index is at least ``MIN_STABILITY`` and its smallest segment holds at
least ``MIN_SEGMENT_SHARE`` of the customers, with k in ``SELECTION_K``. Silhouette scores of
different scalings live in different spaces and are not compared: the scaling is the one with the
highest mean stability over ``SELECTION_K``, and within it the candidate with the highest
silhouette is selected. The same rule picks k for the ``rfm`` features. The HDBSCAN setting of the
selected scaling with the highest silhouette among those with at most ``MAX_NOISE`` noise is kept for
the comparison.

Check on the next quarter, which no clustering saw: how much of the variance of log future spend
and of the future purchase indicator the segment means explain (adjusted eta squared, with 95
percent bootstrap intervals over customers), for the selected k means, k means on RFM only, HDBSCAN,
and the rule based RFM grid.

Outputs under ``artifacts/evaluation/``: ``selection.json``, ``comparison.csv``, ``profiles.csv``,
``segments.parquet`` (segment and map position of every customer), ``jaccard.csv``.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import umap

from customer_segments import clustering
from customer_segments.config import (
    ARTIFACTS_DIR,
    BEHAVIOR,
    BOOTSTRAP_FITS,
    KMEANS_INIT,
    MIN_SEGMENT_SHARE,
    MIN_STABILITY,
    OUTCOME_BOOTSTRAP,
    QUICK,
    SEED,
    SELECTION_K,
    UMAP_MIN_DIST,
    UMAP_NEIGHBORS,
)
from customer_segments.features import SHARES, matrix
from customer_segments.train import kmeans_name

EXPERIMENT = "p09_customer_segments"
MAX_NOISE = 0.3


def setup_mlflow(artifacts_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(artifacts_dir / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(artifacts_dir / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def candidates(grid: pd.DataFrame, stab: pd.DataFrame, feature_set: str, ks: list[int]) -> pd.DataFrame:
    table = grid.merge(stab, on=["feature_set", "scaling", "k"])
    table = table.loc[(table["feature_set"] == feature_set) & table["k"].isin(ks)].copy()
    table["candidate"] = (table["stability_ari"] >= MIN_STABILITY) & (table["min_share"] >= MIN_SEGMENT_SHARE)
    return table


def select(table: pd.DataFrame) -> dict:
    by_scaling = table.groupby("scaling")["stability_ari"].mean().sort_values(ascending=False)
    scaling = by_scaling.index[0]
    within = table.loc[table["scaling"] == scaling]
    pool = within.loc[within["candidate"]]
    rule = "highest silhouette among candidates"
    if pool.empty:
        pool, rule = within.loc[[within["stability_ari"].idxmax()]], "no candidate: most stable k"
    row = pool.sort_values("silhouette", ascending=False).iloc[0]
    return {
        "scaling": scaling,
        "k": int(row["k"]),
        "rule": rule,
        "silhouette": float(row["silhouette"]),
        "stability_ari": float(row["stability_ari"]),
        "min_share": float(row["min_share"]),
        "mean_stability_by_scaling": by_scaling.round(4).to_dict(),
        "candidates": table.loc[table["candidate"], ["scaling", "k"]].to_dict("records"),
    }


def adjusted_eta2(y: np.ndarray, groups: np.ndarray) -> float:
    """1 - (within group variance per degree of freedom) / (total variance per degree of freedom)."""
    n = len(y)
    codes, inverse = np.unique(groups, return_inverse=True)
    g = len(codes)
    if g < 2 or n <= g:
        return 0.0
    means = np.bincount(inverse, weights=y) / np.bincount(inverse)
    ss_within = float(((y - means[inverse]) ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    if ss_total == 0:
        return 0.0
    return 1 - (ss_within / (n - g)) / (ss_total / (n - 1))


def outcome_scores(table: pd.DataFrame, groups: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    targets = {
        "spend": np.log1p(table["future_spend"].to_numpy()),
        "purchase": table["future_purchase"].to_numpy(dtype=float),
    }
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(table), len(table)) for _ in range(n_bootstrap)]
    out = {}
    for name, y in targets.items():
        out[f"eta2_{name}"] = adjusted_eta2(y, groups)
        boot = [adjusted_eta2(y[rows], groups[rows]) for rows in draws]
        out[f"eta2_{name}_low"], out[f"eta2_{name}_high"] = (
            float(v) for v in np.quantile(boot, [0.025, 0.975])
        )
    rates = pd.Series(targets["purchase"]).groupby(groups).mean()
    out["purchase_rate_min"], out["purchase_rate_max"] = float(rates.min()), float(rates.max())
    return out


def pick_density(density: pd.DataFrame, scaling: str) -> str:
    part = density.loc[(density["scaling"] == scaling) & density["method"].str.startswith("hdbscan")]
    ok = part.loc[(part["noise_share"] <= MAX_NOISE) & (part["segments"] >= 2)]
    row = (
        ok.sort_values("silhouette", ascending=False) if len(ok) else part.sort_values("noise_share")
    ).iloc[0]
    return f"{row['method']}_{scaling}"


def order_segments(table: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Segments numbered 1 to k by their share of net spend before the cutoff, largest first."""
    spend = pd.Series(table["monetary"].to_numpy()).groupby(raw).sum().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(spend.index, start=1)}
    return np.array([mapping[v] for v in raw])


def profiles(table: pd.DataFrame, segment: np.ndarray) -> pd.DataFrame:
    frame = table.assign(segment=segment)
    overall_mix = frame[SHARES].mean()
    rows = []
    for seg, part in frame.groupby("segment"):
        mix = part[SHARES].mean()
        lift = (mix / overall_mix.replace(0, np.nan)).fillna(0)
        top = lift.loc[mix >= 0.03].sort_values(ascending=False).head(3)
        row = {
            "segment": int(seg),
            "customers": len(part),
            "customer_share": len(part) / len(frame),
            "spend_share": part["monetary"].sum() / frame["monetary"].sum(),
            **{f"median_{c}": float(part[c].median()) for c in BEHAVIOR if c != "return_share"},
            "mean_return_share": float(part["return_share"].mean()),
            "uk_share": float((part["country"] == "United Kingdom").mean()),
            **{s: float(mix[s]) for s in SHARES},
            "top_categories": ", ".join(f"{c.removeprefix('share_')} x{v:.1f}" for c, v in top.items()),
            "future_purchase_rate": float(part["future_purchase"].mean()),
            "future_spend_mean": float(part["future_spend"].mean()),
            "future_spend_share": float(part["future_spend"].sum() / max(frame["future_spend"].sum(), 1e-9)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run(artifacts: Path, quick: bool) -> None:
    out_dir = artifacts / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(artifacts / "customers.parquet")
    labels = pd.read_parquet(artifacts / "labels.parquet")
    grid = pd.read_csv(artifacts / "kmeans_grid.csv")
    stab = pd.read_csv(artifacts / "stability.csv")
    gap = pd.read_csv(artifacts / "gap.csv")
    density = pd.read_csv(artifacts / "density.csv")
    ks = QUICK["selection_k"] if quick else SELECTION_K
    n_boot = QUICK["outcome_bootstrap"] if quick else OUTCOME_BOOTSTRAP

    main = candidates(grid, stab, "rfm_mix", ks)
    selection = select(main)
    rfm_only = select(candidates(grid, stab, "rfm", ks))
    selection["gap_choice"] = {
        sc: clustering.gap_choice(gap.loc[(gap["feature_set"] == "rfm_mix") & (gap["scaling"] == sc)])
        for sc in sorted(gap["scaling"].unique())
    }
    selection["rfm_only"] = {k: rfm_only[k] for k in ("scaling", "k", "silhouette", "stability_ari")}
    density_name = pick_density(density, selection["scaling"])
    selection["hdbscan"] = density_name

    chosen = kmeans_name("rfm_mix", selection["scaling"], selection["k"])
    compared = {
        "kmeans_rfm_mix": chosen,
        "kmeans_rfm": kmeans_name("rfm", rfm_only["scaling"], rfm_only["k"]),
        "hdbscan": density_name,
        "rfm_grid": "rfm_grid",
    }
    setup_mlflow(artifacts)
    rows = []
    for name, column in compared.items():
        groups = labels[column].astype(str).to_numpy()
        scores = outcome_scores(table, groups, n_boot, SEED)
        row = {"segmentation": name, "column": column, "segments": int(pd.Series(groups).nunique()), **scores}
        rows.append(row)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"column": column, "segments": row["segments"]})
            mlflow.log_metrics({k: v for k, v in scores.items() if np.isfinite(v)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    x = matrix(table, "rfm_mix", selection["scaling"])
    segment = order_segments(table, labels[chosen].to_numpy())
    fits = QUICK["bootstrap_fits"] if quick else BOOTSTRAP_FITS
    jaccard = per_segment_jaccard(x, segment, selection["k"], fits)
    jaccard.to_csv(out_dir / "jaccard.csv", index=False)

    neighbors = QUICK["umap_neighbors"] if quick else UMAP_NEIGHBORS
    embedding = umap.UMAP(
        n_neighbors=neighbors, min_dist=UMAP_MIN_DIST, random_state=SEED, n_jobs=1
    ).fit_transform(x)
    segments = pd.DataFrame(
        {"segment": segment, "umap_x": embedding[:, 0], "umap_y": embedding[:, 1]}, index=table.index
    )
    segments["rfm_grid"] = labels["rfm_grid"].to_numpy()
    segments.to_parquet(out_dir / "segments.parquet")
    profiles(table, segment).merge(jaccard, on="segment").to_csv(out_dir / "profiles.csv", index=False)

    selection["column"] = chosen
    selection["customers"] = len(table)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    print(
        f"selected: k means on rfm_mix, {selection['scaling']} scaling, k = {selection['k']} "
        f"(silhouette {selection['silhouette']:.3f}, stability {selection['stability_ari']:.3f}); "
        f"gap statistic: {selection['gap_choice']}"
    )
    print(
        comparison[["segmentation", "segments", "eta2_spend", "eta2_purchase"]]
        .round(3)
        .to_string(index=False)
    )


def per_segment_jaccard(x: np.ndarray, segment: np.ndarray, k: int, fits: int) -> pd.DataFrame:
    """Mean best Jaccard overlap of each selected segment over bootstrap refits of k means."""
    rng = np.random.default_rng(SEED)
    values = []
    for i in range(fits):
        rows = rng.integers(0, len(x), len(x))
        assigned = clustering.kmeans(x[rows], k, SEED + 1 + i, KMEANS_INIT).predict(x)
        values.append(clustering.best_jaccard(segment, assigned))
    return pd.DataFrame({"segment": np.unique(segment), "jaccard": np.vstack(values).mean(axis=0)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.artifacts, args.quick)


if __name__ == "__main__":
    main()
