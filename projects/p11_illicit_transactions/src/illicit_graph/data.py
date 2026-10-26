"""Read the three Elliptic CSV files, check them, and store a node table and an edge table.

The release is a Kaggle dataset behind the page at ``SOURCE_PAGE``; downloads need a Kaggle login,
so the files are placed by hand in ``data/raw/`` (or in the folder given as ``--source``).
``--source`` may also name the Kaggle zip, which holds the CSV files in a subfolder; they are
extracted to ``data/raw/`` first.

* ``elliptic_txs_features.csv``: no header row; transaction id, time step (1 to 49), 93 local and
  72 aggregated features;
* ``elliptic_txs_classes.csv``: ``txId,class`` with class ``1`` (illicit), ``2`` (licit) or
  ``unknown``;
* ``elliptic_txs_edgelist.csv``: ``txId1,txId2``, one directed edge per payment flow.

Every node gets a split from its time step (see ``SPLIT_STEPS``). Edges are stored as row positions
of the node table, so that the graph libraries need no id lookup.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample
from portfolio_common.config import parse_args

from illicit_graph.config import (
    AGGREGATED,
    AUDIT_PATH,
    CLASS_CODES,
    CLASSES_FILE,
    DATA_DIR,
    DOCUMENTED_COUNTS,
    EDGES_FILE,
    EDGES_PATH,
    FEATURES_FILE,
    ILLICIT,
    LICIT,
    LOCAL,
    NODES_PATH,
    RAW_DIR,
    RAW_FILES,
    SHUTDOWN_STEP,
    SPLIT_STEPS,
    SPLITS,
    TIME_STEPS,
    UNKNOWN,
)

FEATURES = LOCAL + AGGREGATED


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def extract(archive: Path, target: Path) -> Path:
    """Extract the three CSV files from a zip, wherever they sit in it, into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            name = Path(member.filename).name
            if name in RAW_FILES and not member.is_dir():
                with handle.open(member) as source, (target / name).open("wb") as out:
                    while block := source.read(1 << 20):
                        out.write(block)
    return target


def check_present(source: Path) -> None:
    missing = [name for name in RAW_FILES if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{source} lacks {missing}; download the Elliptic dataset from Kaggle (see the README)"
        )


def read_features(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype="float64")
    expected = 2 + len(FEATURES)
    if raw.shape[1] != expected:
        raise SchemaError(f"{path.name} has {raw.shape[1]} columns, expected {expected}")
    raw.columns = ["tx_id", "time_step", *FEATURES]
    if raw[["tx_id", "time_step"]].isna().any().any():
        raise SchemaError(f"{path.name} has empty transaction ids or time steps")
    steps = raw["time_step"]
    if not steps.between(1, TIME_STEPS).all() or not (steps == steps.round()).all():
        raise SchemaError(f"{path.name} has time steps outside 1 to {TIME_STEPS}")
    out = pd.DataFrame({"tx_id": raw["tx_id"].astype("int64"), "time_step": steps.astype("int8")})
    return pd.concat([out, raw[FEATURES].astype("float32")], axis=1)


def read_classes(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    if list(raw.columns) != ["txId", "class"]:
        raise SchemaError(f"{path.name} has columns {list(raw.columns)}, expected ['txId', 'class']")
    codes = raw["class"].str.strip().str.lower()
    unknown_codes = sorted(set(codes) - set(CLASS_CODES))
    if unknown_codes:
        raise SchemaError(f"{path.name} has class values {unknown_codes}")
    return pd.DataFrame(
        {"tx_id": raw["txId"].astype("int64"), "label": codes.map(CLASS_CODES).astype("int8")}
    )


def read_edges(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype="int64")
    raw.columns = [c.strip() for c in raw.columns]
    if list(raw.columns) != ["txId1", "txId2"]:
        raise SchemaError(f"{path.name} has columns {list(raw.columns)}, expected ['txId1', 'txId2']")
    return raw.rename(columns={"txId1": "src_id", "txId2": "dst_id"})


def split_of(time_step: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=time_step.index, dtype=object)
    for name, steps in SPLIT_STEPS.items():
        out[time_step.isin(list(steps))] = name
    return pd.Categorical(out, categories=SPLITS)


def build(
    features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    issues = {
        "duplicate_node_ids": int(features["tx_id"].duplicated().sum()),
        "duplicate_class_ids": int(classes["tx_id"].duplicated().sum()),
        "edge_rows": len(edges),
    }
    if issues["duplicate_node_ids"]:
        raise SchemaError(f"{FEATURES_FILE} repeats {issues['duplicate_node_ids']} transaction ids")
    classes = classes.drop_duplicates("tx_id")
    nodes = features.merge(classes, on="tx_id", how="left", validate="one_to_one")
    issues["nodes_without_class_row"] = int(nodes["label"].isna().sum())
    issues["class_rows_without_node"] = int((~classes["tx_id"].isin(features["tx_id"])).sum())
    nodes["label"] = nodes["label"].fillna(UNKNOWN).astype("int8")
    nodes = nodes.sort_values(["time_step", "tx_id"], kind="stable").reset_index(drop=True)
    nodes.insert(2, "split", split_of(nodes["time_step"]))
    nodes = nodes[["tx_id", "time_step", "split", "label", *FEATURES]]

    position = pd.Series(np.arange(len(nodes)), index=nodes["tx_id"].to_numpy())
    known = edges["src_id"].isin(position.index) & edges["dst_id"].isin(position.index)
    issues["edges_with_unknown_node"] = int((~known).sum())
    edges = edges.loc[known]
    issues["self_loops"] = int((edges["src_id"] == edges["dst_id"]).sum())
    issues["duplicate_edges"] = int(edges.duplicated().sum())
    edges = edges.loc[edges["src_id"] != edges["dst_id"]].drop_duplicates()
    out = pd.DataFrame(
        {
            "src": position.loc[edges["src_id"]].to_numpy(),
            "dst": position.loc[edges["dst_id"]].to_numpy(),
            "src_id": edges["src_id"].to_numpy(),
            "dst_id": edges["dst_id"].to_numpy(),
        }
    )
    steps = nodes["time_step"].to_numpy()
    issues["edges_across_time_steps"] = int((steps[out["src"]] != steps[out["dst"]]).sum())
    return nodes, out, issues


def audit(nodes: pd.DataFrame, edges: pd.DataFrame, issues: dict) -> dict:
    found = {
        "nodes": len(nodes),
        "edges": issues["edge_rows"],
        "illicit": int((nodes["label"] == ILLICIT).sum()),
        "licit": int((nodes["label"] == LICIT).sum()),
    }
    degree = np.bincount(np.concatenate([edges["src"], edges["dst"]]), minlength=len(nodes))
    by_split = {}
    for name in SPLITS:
        part = nodes.loc[nodes["split"] == name, "label"]
        by_split[name] = {
            "nodes": len(part),
            "illicit": int((part == ILLICIT).sum()),
            "licit": int((part == LICIT).sum()),
            "unknown": int((part == UNKNOWN).sum()),
        }
    labelled = nodes["label"] != UNKNOWN
    return {
        "counts_found": found,
        "counts_documented": DOCUMENTED_COUNTS,
        "counts_match_documented": found == DOCUMENTED_COUNTS,
        "labelled_share": round(float(labelled.mean()), 4),
        "illicit_share_of_labelled": round(float((nodes.loc[labelled, "label"] == ILLICIT).mean()), 4),
        "isolated_nodes": int((degree == 0).sum()),
        "time_steps": int(nodes["time_step"].nunique()),
        "by_split": by_split,
        **issues,
    }


def prepare(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if source.is_file() and zipfile.is_zipfile(source):
        source = extract(source, RAW_DIR)
    check_present(source)
    nodes, edges, issues = build(
        read_features(source / FEATURES_FILE),
        read_classes(source / CLASSES_FILE),
        read_edges(source / EDGES_FILE),
    )
    return nodes, edges, audit(nodes, edges, issues)


def sample_steps(
    nodes: pd.DataFrame, edges: pd.DataFrame, share: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep whole time step graphs, stratified by split, and renumber the edge positions.

    No edge joins two time steps, so every kept graph is complete. The steps just before and at
    the dark market shutdown are always kept, so the test period has steps on both sides of it.
    """
    keep = sample.select(
        nodes["time_step"], share, seed, strata=nodes["split"], always=[SHUTDOWN_STEP - 1, SHUTDOWN_STEP]
    )
    position = np.full(len(nodes), -1)
    position[keep] = np.arange(int(keep.sum()))
    kept_edges = edges[keep[edges["src"].to_numpy()] & keep[edges["dst"].to_numpy()]].copy()
    kept_edges["src"] = position[kept_edges["src"].to_numpy()]
    kept_edges["dst"] = position[kept_edges["dst"].to_numpy()]
    return nodes[keep].reset_index(drop=True), kept_edges.reset_index(drop=True)


def write(nodes: pd.DataFrame, edges: pd.DataFrame, report: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(data_dir / NODES_PATH.name, index=False)
    edges.to_parquet(data_dir / EDGES_PATH.name, index=False)
    (data_dir / AUDIT_PATH.name).write_text(json.dumps(report, indent=2))


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(data_dir / NODES_PATH.name), pd.read_parquet(data_dir / EDGES_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DIR, help="folder (or zip) with the CSV files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    sample.add_arguments(parser, "time step graphs, stratified by split")
    args = parse_args(parser, __file__)
    nodes, edges, report = prepare(args.source)
    # The audit describes the whole source; the sample is drawn after it.
    n_steps = nodes["time_step"].nunique()
    nodes, edges = sample_steps(nodes, edges, args.sample, args.sample_seed)
    write(nodes, edges, report, args.data_dir)
    steps = nodes["time_step"].nunique()
    sample.record(args.data_dir, args.sample, args.sample_seed, "time step graph", n_steps, steps)
    found = report["counts_found"]
    print(
        f"{found['nodes']:,} transactions, {found['edges']:,} edges, {found['illicit']:,} illicit, "
        f"{found['licit']:,} licit; counts match the documented ones: {report['counts_match_documented']}"
    )
    print(f"edges across time steps: {report['edges_across_time_steps']}")
    for name, row in report["by_split"].items():
        print(f"  {name:<10} " + "  ".join(f"{k} {v:>7,}" for k, v in row.items()))


if __name__ == "__main__":
    main()
