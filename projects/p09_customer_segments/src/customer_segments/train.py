"""Build the customer table and fit every clustering of the comparison.

Stages, each stored under ``artifacts/`` and skipped when its file exists (``--force`` refits):

* ``customers.parquet``: features up to the cutoff and the outcome after it (``features.py``);
* ``kmeans_grid.csv``: k means for each feature set, scaling and k, with quality scores;
* ``gap.csv``: the gap statistic for the same grid;
* ``stability.csv``: bootstrap stability for the same grid;
* ``density.csv``: DBSCAN and HDBSCAN on the ``rfm_mix`` features for each scaling but ``raw``;
* ``labels.parquet``: the segment of every customer under every clustering, and the RFM grid.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from customer_segments import clustering
from customer_segments.config import (
    ARTIFACTS_DIR,
    BOOTSTRAP_FITS,
    DATA_DIR,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_QUANTILES,
    FEATURE_SETS,
    GAP_REFERENCES,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_SIZES,
    K_RANGE,
    KMEANS_INIT,
    OUTCOME_DAYS,
    QUICK,
    SCALINGS,
    SEED,
)
from customer_segments.data import load
from customer_segments.features import customer_table, cutoff_date, matrix

BOOTSTRAP_INIT = 3  # restarts per bootstrap refit; the full fits use KMEANS_INIT
DENSITY_SCALINGS = [s for s in SCALINGS if s != "raw"]


def kmeans_name(feature_set: str, scaling: str, k: int) -> str:
    return f"kmeans_{feature_set}_{scaling}_k{k}"


def settings(quick: bool) -> dict:
    if quick:
        return {"k_range": QUICK["k_range"], "gap": QUICK["gap_references"], "fits": QUICK["bootstrap_fits"]}
    return {"k_range": K_RANGE, "gap": GAP_REFERENCES, "fits": BOOTSTRAP_FITS}


def stage(path: Path, force: bool) -> bool:
    """True when the stage has to run."""
    if path.exists() and not force:
        print(f"  {path.name}: stored, skipped")
        return False
    return True


def build_customers(data_dir: Path, artifacts: Path, force: bool) -> pd.DataFrame:
    path = artifacts / "customers.parquet"
    if stage(path, force):
        lines = load(data_dir)
        cutoff = cutoff_date(lines)
        table = customer_table(lines, cutoff)
        table.to_parquet(path)
        info = {
            "cutoff": str(cutoff.date()),
            "outcome_days": OUTCOME_DAYS,
            "last_date": str(lines["invoice_date"].max()),
            "customers": len(table),
            "customers_with_purchase_before_cutoff": int(
                lines.loc[lines["is_product"] & (lines["invoice_date"] < cutoff), "customer_id"].nunique()
            ),
            "future_purchase_share": round(float(table["future_purchase"].mean()), 4),
        }
        (artifacts / "customers.json").write_text(json.dumps(info, indent=2))
        print(f"  {len(table):,} customers at the cutoff {info['cutoff']}")
    return pd.read_parquet(path)


def run(data_dir: Path, artifacts: Path, quick: bool, force: bool) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    conf = settings(quick)
    table = build_customers(data_dir, artifacts, force)
    spaces = {(fs, sc): matrix(table, fs, sc) for fs in FEATURE_SETS for sc in SCALINGS}
    labels_path = artifacts / "labels.parquet"
    labels = (
        pd.read_parquet(labels_path)
        if labels_path.exists() and not force
        else pd.DataFrame(index=table.index)
    )

    path = artifacts / "kmeans_grid.csv"
    if stage(path, force):
        frames = []
        for (fs, sc), x in spaces.items():
            started = time.time()
            grid, fitted = clustering.kmeans_grid(x, conf["k_range"], SEED, KMEANS_INIT)
            frames.append(grid.assign(feature_set=fs, scaling=sc))
            for k, values in fitted.items():
                labels[kmeans_name(fs, sc, k)] = values
            print(f"  k means {fs} {sc}: {time.time() - started:.1f} s")
        labels.to_parquet(labels_path)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)

    path = artifacts / "gap.csv"
    if stage(path, force):
        frames = [
            clustering.gap_statistic(x, conf["k_range"], conf["gap"], SEED).assign(feature_set=fs, scaling=sc)
            for (fs, sc), x in spaces.items()
        ]
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)

    path = artifacts / "stability.csv"
    if stage(path, force):
        rows = []
        for (fs, sc), x in spaces.items():
            started = time.time()
            for k in conf["k_range"]:
                rows.append(
                    {"feature_set": fs, "scaling": sc}
                    | clustering.stability(x, k, conf["fits"], SEED, BOOTSTRAP_INIT)
                )
            print(f"  stability {fs} {sc}: {time.time() - started:.1f} s")
        pd.DataFrame(rows).to_csv(path, index=False)

    path = artifacts / "density.csv"
    if stage(path, force):
        frames = []
        for sc in DENSITY_SCALINGS:
            x = spaces[("rfm_mix", sc)]
            db, db_labels = clustering.dbscan_grid(x, DBSCAN_MIN_SAMPLES, DBSCAN_QUANTILES, SEED)
            hd, hd_labels = clustering.hdbscan_grid(x, HDBSCAN_SIZES, HDBSCAN_MIN_SAMPLES, SEED)
            frames += [db.assign(scaling=sc), hd.assign(scaling=sc)]
            for name, values in (db_labels | hd_labels).items():
                labels[f"{name}_{sc}"] = values
        labels.to_parquet(labels_path)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)

    labels["rfm_grid"] = clustering.rfm_grid(table)
    labels.to_parquet(labels_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="small grid and few refits, for tests")
    parser.add_argument("--force", action="store_true", help="refit stages that are stored")
    args = parser.parse_args()
    run(args.data_dir, args.artifacts, args.quick, args.force)


if __name__ == "__main__":
    main()
