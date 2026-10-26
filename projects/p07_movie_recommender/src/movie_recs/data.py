"""Read the MovieLens 32M zip, check its layout, and store Parquet tables and the time split.

The GroupLens host may be unreachable from a build environment, so the zip can
also be given as a local file (``--source``). Its MD5 is checked against the
value that GroupLens publishes in ``ml-32m.zip.md5``; ``--md5 any`` skips the
check (for the synthetic archive of the smoke run).
"""

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample
from portfolio_common.config import parse_args

from movie_recs.config import (
    ARCHIVE_ROOT,
    DATA_DIR,
    EXPECTED_MOVIES,
    EXPECTED_RATINGS,
    EXPECTED_USERS,
    GENRES,
    MOVIES_PATH,
    RATINGS_PATH,
    RAW_DIR,
    RAW_NAME,
    SOURCE_FILES,
    SOURCE_MD5,
    SOURCE_URL,
    SPLIT_PATH,
    TAGS_PATH,
    TEST_QUANTILE,
    VALIDATION_QUANTILE,
)

RATING_VALUES = np.arange(0.5, 5.01, 0.5)
YEAR = re.compile(r"\((\d{4})\)\s*$")


class SchemaError(ValueError):
    """A source file does not match the documented layout."""


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch the zip (about 240 MB) unless it is present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_NAME
    if not target.exists():
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(SOURCE_URL, timeout=600) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(target)
    return target


def read_archive(path: Path) -> dict[str, pd.DataFrame]:
    """Read ratings, movies and tags from the zip."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        wanted = {key: f"{ARCHIVE_ROOT}/{name}" for key, name in SOURCE_FILES.items()}
        missing = sorted(set(wanted.values()) - names)
        if missing:
            raise SchemaError(f"archive lacks {missing}")
        tables = {}
        with archive.open(wanted["ratings"]) as handle:
            tables["ratings"] = pd.read_csv(
                handle,
                dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
                engine="pyarrow",
            )
        with archive.open(wanted["movies"]) as handle:
            tables["movies"] = pd.read_csv(handle, dtype={"movieId": "int32"})
        with archive.open(wanted["tags"]) as handle:
            tables["tags"] = pd.read_csv(handle, dtype={"userId": "int32", "movieId": "int32"})
    return tables


def check(tables: dict[str, pd.DataFrame], expect_counts: bool) -> None:
    ratings, movies, tags = tables["ratings"], tables["movies"], tables["tags"]
    for name, frame, columns in (
        ("ratings", ratings, ["userId", "movieId", "rating", "timestamp"]),
        ("movies", movies, ["movieId", "title", "genres"]),
        ("tags", tags, ["userId", "movieId", "tag", "timestamp"]),
    ):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise SchemaError(f"{name} lacks {missing}")
    if ratings.isna().any().any():
        raise SchemaError("ratings have missing values")
    if not np.isin(ratings["rating"].to_numpy(), RATING_VALUES).all():
        raise SchemaError("ratings outside 0.5 to 5.0 in half stars")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise SchemaError("a user rates the same movie twice")
    if movies["movieId"].duplicated().any():
        raise SchemaError("movies repeat a movieId")
    if not ratings["movieId"].isin(movies["movieId"]).all():
        raise SchemaError("ratings name a movie that movies.csv lacks")
    if expect_counts:
        found = (len(ratings), ratings["userId"].nunique(), len(movies))
        expected = (EXPECTED_RATINGS, EXPECTED_USERS, EXPECTED_MOVIES)
        if found != expected:
            raise SchemaError(f"ratings, users, movies: expected {expected}, found {found}")


def movie_table(movies: pd.DataFrame) -> pd.DataFrame:
    """One row per movie: title, release year (from the title) and one 0/1 column per genre."""
    out = pd.DataFrame({"item": movies["movieId"].astype("int32"), "title": movies["title"].str.strip()})
    out["year"] = movies["title"].str.extract(YEAR, expand=False).astype("float32")
    genres = movies["genres"].fillna("").str.split("|")
    for genre in GENRES:
        out[genre] = genres.map(lambda g, genre=genre: genre in g).astype("int8")
    return out


def time_split(ts: np.ndarray) -> dict:
    """Start of the validation and test windows: quantiles of the rating times."""
    validation_start = int(np.quantile(ts, VALIDATION_QUANTILE))
    test_start = int(np.quantile(ts, TEST_QUANTILE))
    end = int(ts.max())
    if not validation_start < test_start <= end:
        raise SchemaError("rating times are too few to split")
    return {"validation_start": validation_start, "test_start": test_start, "end": end}


def normalize(
    tables: dict[str, pd.DataFrame],
    expect_counts: bool = True,
    share: float = sample.FULL,
    seed: int = sample.DEFAULT_SEED,
) -> dict:
    """Check the release, keep a share of whole users (all their ratings) and split by time.

    The release counts are checked before the sample; the time split is computed on the sample.
    """
    check(tables, expect_counts)
    ratings = tables["ratings"].rename(columns={"userId": "user", "movieId": "item", "timestamp": "ts"})
    ratings = ratings[sample.select(ratings["user"], share, seed)]
    ratings = ratings.sort_values(["ts", "user", "item"], kind="stable").reset_index(drop=True)
    tags = tables["tags"].rename(columns={"movieId": "item", "timestamp": "ts"})
    tags = tags.assign(tag=tags["tag"].astype(str).str.strip().str.lower())
    tags = tags.loc[tags["tag"] != "", ["item", "tag", "ts"]].reset_index(drop=True)
    return {
        "ratings": ratings,
        "movies": movie_table(tables["movies"]),
        "tags": tags,
        "split": time_split(ratings["ts"].to_numpy()),
    }


def write(processed: dict, data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    processed["ratings"].to_parquet(data_dir / RATINGS_PATH.name, index=False)
    processed["movies"].to_parquet(data_dir / MOVIES_PATH.name, index=False)
    processed["tags"].to_parquet(data_dir / TAGS_PATH.name, index=False)
    (data_dir / SPLIT_PATH.name).write_text(json.dumps(processed["split"], indent=2))


def load(data_dir: Path = DATA_DIR) -> dict:
    return {
        "ratings": pd.read_parquet(data_dir / RATINGS_PATH.name),
        "movies": pd.read_parquet(data_dir / MOVIES_PATH.name),
        "tags": pd.read_parquet(data_dir / TAGS_PATH.name),
        "split": json.loads((data_dir / SPLIT_PATH.name).read_text()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="a local copy of ml-32m.zip instead of the download")
    parser.add_argument("--md5", default=SOURCE_MD5, help="expected MD5 of the zip; 'any' skips the check")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-count-check", action="store_true", help="skip the release size checks")
    sample.add_arguments(parser, "users, each with all their ratings")
    args = parse_args(parser, __file__)
    archive = args.source if args.source else download()
    md5 = digest(archive, "md5")
    if args.md5 != "any" and md5 != args.md5.strip().lower():
        raise SchemaError(f"{archive} has MD5 {md5}, expected {args.md5}")
    tables = read_archive(archive)
    n_users = tables["ratings"]["userId"].nunique()
    processed = normalize(tables, not args.no_count_check, args.sample, args.sample_seed)
    write(processed, args.data_dir)
    ratings, split = processed["ratings"], processed["split"]
    sample.record(args.data_dir, args.sample, args.sample_seed, "user", n_users, ratings["user"].nunique())
    print(
        f"{len(ratings):,} ratings of {ratings['user'].nunique():,} users on {ratings['item'].nunique():,} "
        f"movies written to {args.data_dir}; zip MD5 {md5}"
    )
    for key, value in split.items():
        print(f"  {key}: {pd.Timestamp(value, unit='s')}")


if __name__ == "__main__":
    main()
