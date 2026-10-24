"""Read the UNSW-NB15 CSV files, check them, and store one table of flows with a split column.

The release is a shared folder behind the page at ``SOURCE_PAGE``; it has no stable direct file
links, so the files are placed by hand in ``data/raw/`` (or in the folder given as ``--source``).
Two layouts are read:

* ``full``: ``UNSW-NB15_1.csv`` to ``UNSW-NB15_4.csv`` without a header row, with the column names
  from ``NUSW-NB15_features.csv`` (or the built in list when that file is absent);
* ``partition``: ``UNSW_NB15_training-set.csv`` and ``UNSW_NB15_testing-set.csv`` with a header row.

When both are present the full records are used. Every column the model uses is converted to a
number; cells that do not convert are counted in the audit and set to 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flow_anomaly.config import (
    ATTACK_NAMES,
    AUDIT_PATH,
    CATEGORICAL,
    DATA_DIR,
    DOCUMENTED_COUNTS,
    FEATURES_FILE,
    FLOWS_PATH,
    FULL_COLUMNS,
    FULL_FILES,
    FULL_RENAME,
    NORMAL,
    NUMERIC,
    PARTITION_FILES,
    PARTITION_VALIDATION_SHARE,
    RAW_DIR,
    SEED,
    SPLIT_SHARES,
)

SPLITS = ["train", "validation", "test"]


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def normalize_name(name: str) -> str:
    return "".join(str(name).lower().split())


def detect_layout(source: Path) -> str:
    if all((source / name).exists() for name in FULL_FILES):
        return "full"
    if all((source / name).exists() for name in PARTITION_FILES.values()):
        return "partition"
    raise FileNotFoundError(
        f"{source} holds neither {FULL_FILES} nor {list(PARTITION_FILES.values())}; "
        "download them from the UNSW-NB15 page (see the README)"
    )


def full_column_names(source: Path) -> list[str]:
    path = source / FEATURES_FILE
    if not path.exists():
        return list(FULL_COLUMNS)
    features = pd.read_csv(path, encoding="latin-1")
    features.columns = [normalize_name(c) for c in features.columns]
    if "name" not in features.columns:
        raise SchemaError(f"{FEATURES_FILE} lacks a Name column")
    names = [normalize_name(n) for n in features["name"]]
    if len(names) != len(FULL_COLUMNS):
        raise SchemaError(f"{FEATURES_FILE} names {len(names)} columns, expected {len(FULL_COLUMNS)}")
    return names


def read_full(source: Path) -> pd.DataFrame:
    names = full_column_names(source)
    parts = []
    for name in FULL_FILES:
        part = pd.read_csv(source / name, header=None, low_memory=False, encoding="latin-1")
        if part.shape[1] != len(names):
            raise SchemaError(f"{name} has {part.shape[1]} columns, expected {len(names)}")
        part.columns = names
        part["source_file"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def read_partition(source: Path) -> pd.DataFrame:
    parts = []
    for role, name in PARTITION_FILES.items():
        part = pd.read_csv(source / name, low_memory=False, encoding="latin-1")
        part.columns = [normalize_name(c) for c in part.columns]
        part["source_file"] = name
        part["source_role"] = role
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def attack_names(raw: pd.Series, label: np.ndarray) -> tuple[pd.Series, dict]:
    """Canonical attack type: stripped, one spelling per type, Normal for label 0."""
    text = raw.fillna("").astype(str).str.strip()
    mapped = text.str.lower().map(ATTACK_NAMES)
    out = mapped.fillna(text)
    out = out.where(out != "", NORMAL)
    issues = {
        "label_0_with_attack_type": int(((label == 0) & (out != NORMAL)).sum()),
        "label_1_without_attack_type": int(((label == 1) & (out == NORMAL)).sum()),
    }
    # The label decides: a flow labeled normal is normal, an attack without a type is "Unknown".
    out = out.where(label == 1, NORMAL)
    out = out.where(~((label == 1) & (out == NORMAL)), "Unknown")
    return out, issues


def normalize(raw: pd.DataFrame, layout: str) -> tuple[pd.DataFrame, dict]:
    """Canonical columns: the numeric features, the categorical ones, label, attack_cat, source info."""
    frame = raw.rename(columns=FULL_RENAME) if layout == "full" else raw
    missing = [c for c in NUMERIC + CATEGORICAL + ["label", "attack_cat"] if c not in frame.columns]
    if missing:
        raise SchemaError(f"{layout} files lack {missing}")
    out = pd.DataFrame(index=frame.index)
    not_numeric = {}
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = int(values.isna().sum())
        if bad:
            not_numeric[column] = bad
        out[column] = values.fillna(0).astype("float64")
    for column in CATEGORICAL:
        out[column] = frame[column].fillna("-").astype(str).str.strip().str.lower()
    label = pd.to_numeric(frame["label"], errors="coerce")
    if label.isna().any() or not label.isin([0, 1]).all():
        raise SchemaError("label holds values other than 0 and 1")
    out["label"] = label.astype("int8")
    out["attack_cat"], issues = attack_names(frame["attack_cat"], out["label"].to_numpy())
    out["source_file"] = frame["source_file"].astype(str)
    if "stime" in frame.columns:
        out["stime"] = pd.to_numeric(frame["stime"], errors="coerce").fillna(0).astype("int64")
    if "source_role" in frame.columns:
        out["source_role"] = frame["source_role"].astype(str)
    issues["not_numeric_cells"] = not_numeric
    return out, issues


def stratified_split(strata: pd.Series, shares: dict[str, float], seed: int) -> np.ndarray:
    """Assign every row to a split, so that each stratum is divided in the given shares."""
    rng = np.random.default_rng(seed)
    names = list(shares)
    edges = np.cumsum([shares[n] for n in names])
    if not np.isclose(edges[-1], 1.0):
        raise ValueError("split shares must add up to 1")
    out = np.empty(len(strata), dtype=object)
    codes = strata.to_numpy()
    for value in pd.unique(codes):
        rows = np.flatnonzero(codes == value)
        rows = rows[rng.permutation(len(rows))]
        bounds = np.round(edges * len(rows)).astype(int)
        start = 0
        for name, end in zip(names, bounds, strict=True):
            out[rows[start:end]] = name
            start = end
    return out


def assign_split(flows: pd.DataFrame, layout: str, seed: int = SEED) -> pd.Series:
    if layout == "full":
        return pd.Series(stratified_split(flows["attack_cat"], SPLIT_SHARES, seed), index=flows.index)
    split = pd.Series("test", index=flows.index, dtype=object)
    train = flows["source_role"] == "train"
    shares = {"train": 1 - PARTITION_VALIDATION_SHARE, "validation": PARTITION_VALIDATION_SHARE}
    split[train] = stratified_split(flows.loc[train, "attack_cat"], shares, seed)
    return split


def audit(flows: pd.DataFrame, layout: str, issues: dict) -> dict:
    counts = (
        flows.groupby(["attack_cat", "split"]).size().unstack("split", fill_value=0).reindex(columns=SPLITS)
    )
    found = {"rows": len(flows), "attacks": int(flows["label"].sum())}
    if layout == "partition":
        found = {role: int((flows["source_role"] == role).sum()) for role in PARTITION_FILES}
    documented = DOCUMENTED_COUNTS[layout]
    return {
        "layout": layout,
        "rows": len(flows),
        "attacks": int(flows["label"].sum()),
        "attack_share": round(float(flows["label"].mean()), 5),
        "duplicate_rows": int(flows.drop(columns=["source_file"]).duplicated().sum()),
        "counts_found": found,
        "counts_documented": documented,
        "counts_match_documented": found == documented,
        "by_type_and_split": {k: {s: int(v) for s, v in row.items()} for k, row in counts.iterrows()},
        "files": sorted(flows["source_file"].unique().tolist()),
        **issues,
    }


def prepare(source: Path, layout: str | None = None) -> tuple[pd.DataFrame, dict]:
    layout = layout or detect_layout(source)
    raw = read_full(source) if layout == "full" else read_partition(source)
    flows, issues = normalize(raw, layout)
    flows["split"] = assign_split(flows, layout).astype("category")
    return flows, audit(flows, layout, issues)


def write(flows: pd.DataFrame, report: dict, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(data_dir / FLOWS_PATH.name, index=False)
    (data_dir / AUDIT_PATH.name).write_text(json.dumps(report, indent=2))


def load(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / FLOWS_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DIR, help="folder with the UNSW-NB15 CSV files")
    parser.add_argument("--layout", choices=["full", "partition"], default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    flows, report = prepare(args.source, args.layout)
    write(flows, report, args.data_dir)
    print(
        f"{report['layout']}: {report['rows']:,} flows, {report['attacks']:,} attacks "
        f"({report['attack_share']:.1%}); counts match the documented ones: "
        f"{report['counts_match_documented']}"
    )
    for name, row in report["by_type_and_split"].items():
        print(f"  {name:<15} " + "  ".join(f"{s} {row[s]:>9,}" for s in SPLITS))


if __name__ == "__main__":
    main()
