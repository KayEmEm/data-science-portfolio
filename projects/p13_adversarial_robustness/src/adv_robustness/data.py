"""Read the data of the three target projects with their own loaders, draw the sample, and store it.

* credit: the UCI spreadsheet of project 1 (or a client table in its schema), the feature SQL of
  project 1, a sample of clients stratified by default;
* flows: the UNSW-NB15 files of project 8, its split, a sample of flows stratified by attack type and
  split;
* images: the RDD2022 country zips of project 6, its block split, a sample of blocks.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample

from adv_robustness.config import (
    CREDIT_SOURCE,
    DATA_DIR,
    FLOWS_SOURCE,
    IMAGE_COUNTRIES,
    IMAGES_SOURCE,
    PARTS,
)


class SourceError(FileNotFoundError):
    """A source folder lacks the files a part needs."""


def credit(source: Path, out: Path, share: float, seed: int) -> dict:
    from loan_default.config import ID_COL, TARGET_COL
    from loan_default.data import find_spreadsheet, normalize, read_spreadsheet
    from loan_default.features import build_features

    out.mkdir(parents=True, exist_ok=True)
    tables = sorted(source.glob("*.parquet"))
    if tables:
        clients = normalize(pd.read_parquet(tables[0]))  # a table in the source schema (smoke run)
    else:
        raw = out / "raw"
        raw.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix.lower() in {".zip", ".xls", ".xlsx"}:
                shutil.copy2(path, raw / path.name)
        try:
            clients = read_spreadsheet(find_spreadsheet(raw))
        except FileNotFoundError as error:
            raise SourceError(f"no UCI zip, spreadsheet or client table in {source}") from error
    n_full = len(clients)
    clients = clients[sample.select(clients[ID_COL], share, seed, strata=clients[TARGET_COL])]
    clients = clients.reset_index(drop=True)
    clients.to_parquet(out / "clients.parquet", index=False)
    build_features(out / "clients.parquet").to_parquet(out / "features.parquet", index=False)
    return sample.record(out, share, seed, "client", n_full, len(clients))


def flows(source: Path, out: Path, share: float, seed: int) -> dict:
    from flow_anomaly.data import prepare, write

    if not source.exists():
        raise SourceError(f"no UNSW-NB15 files in {source}")
    table, report = prepare(source)
    n_full = len(table)
    strata = table["attack_cat"].astype(str) + "_" + table["split"].astype(str)
    table = table[sample.select(pd.Series(np.arange(n_full)), share, seed, strata=strata)]
    write(table.reset_index(drop=True), report, out)
    return sample.record(out, share, seed, "flow", n_full, len(table))


def images(source: Path, out: Path, share: float, seed: int, block: int | None = None) -> dict:
    from road_damage.config import SPLIT_BLOCK
    from road_damage.data import build

    if not any(source.glob("*.zip")) and not any(source.iterdir()):
        raise SourceError(f"no RDD2022 zips in {source}")
    counts = build(source, out, IMAGE_COUNTRIES, block or SPLIT_BLOCK, share, seed)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    return json.loads((out / sample.RECORD_NAME).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--credit-source", type=Path, default=CREDIT_SOURCE)
    parser.add_argument("--flows-source", type=Path, default=FLOWS_SOURCE)
    parser.add_argument("--images-source", type=Path, default=IMAGES_SOURCE)
    parser.add_argument("--block", type=int, default=None, help="images per split block (project 6: 25)")
    sample.add_arguments(parser, "clients, flows and image blocks, each stratified as in its project")
    args = parser.parse_args()
    steps = {
        "credit": lambda d: credit(args.credit_source, d, args.sample, args.sample_seed),
        "flows": lambda d: flows(args.flows_source, d, args.sample, args.sample_seed),
        "images": lambda d: images(args.images_source, d, args.sample, args.sample_seed, args.block),
    }
    for part in args.parts:
        record = steps[part](args.data_dir / part)
        print(
            f"{part}: kept {record['units_after']:,} of {record['units_before']:,} (unit: {record['unit']})"
        )


if __name__ == "__main__":
    main()
