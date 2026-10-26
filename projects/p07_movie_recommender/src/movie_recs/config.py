"""Paths, source expectations, split and model settings for project 7."""

from dataclasses import dataclass
from pathlib import Path

from portfolio_common.config import settings as yaml_settings

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

# MovieLens 32M, the latest stable full release of GroupLens (May 2024).
# The GroupLens terms forbid redistribution, so the zip is never committed.
SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
# The MD5 that GroupLens publishes in ml-32m.zip.md5; it pins the exact archive this project was run on.
SOURCE_MD5 = "d472be332d4daa821edc399621853b57"
RAW_NAME = "ml-32m.zip"
ARCHIVE_ROOT = "ml-32m"
SOURCE_FILES = {
    "ratings": "ratings.csv",  # userId, movieId, rating, timestamp
    "movies": "movies.csv",  # movieId, title, genres
    "tags": "tags.csv",  # userId, movieId, tag, timestamp
}
# Counts stated in the README of the release.
EXPECTED_RATINGS = 32_000_204
EXPECTED_USERS = 200_948
EXPECTED_MOVIES = 87_585
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
SEED = 42

# Processed tables written by ``make data``.
RATINGS_PATH = DATA_DIR / "ratings.parquet"  # user, item, rating, ts (seconds), sorted by ts
MOVIES_PATH = DATA_DIR / "movies.parquet"  # item, title, year, genre columns
TAGS_PATH = DATA_DIR / "tags.parquet"  # item, tag, ts
SPLIT_PATH = DATA_DIR / "split.json"

# Time based split: the last 2 percent of ratings by time are the test window,
# the 2 percent before them the validation window. Every model sees only the
# ratings before the start of the window it is scored on.
VALIDATION_QUANTILE = 0.96
TEST_QUANTILE = 0.98
# A rating of 4 or more in the window counts as a movie the user liked: these
# are the movies a recommendation should have found.
POSITIVE_RATING = 4.0
K = 10
N_CANDIDATES = 100  # per retrieval model, passed to the ranking model
RECENT_DAYS = 90  # window of the recent popularity baseline and feature


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class ALSSettings:
    factors: int = 128
    regularization: float = 0.05
    alpha: float = 1.0
    iterations: int = 15


# Searched on the validation window; the best setting is refit for the test window.
ALS_GRID = [
    ALSSettings(factors=64, regularization=0.05),
    ALSSettings(factors=128, regularization=0.05),
    ALSSettings(factors=128, regularization=0.5),
    ALSSettings(factors=256, regularization=0.5),
]


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class TwoTowerSettings:
    dim: int = 64
    history: int = 50  # most recent movies that describe a user
    targets_per_user: int = 20  # training examples sampled per user
    batch_size: int = 2048
    epochs: int = 6
    lr: float = 2e-3
    weight_decay: float = 1e-6
    temperature: float = 0.05
    id_dropout: float = 0.1  # share of item ids hidden in training, so that features alone carry new movies
    n_tags: int = 1000  # most frequent tags used as item features
    threads: int = 4


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class RankerSettings:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8


QUICK_ALS_GRID = [ALSSettings(factors=16, regularization=0.05, iterations=8)]
QUICK_TWO_TOWER = TwoTowerSettings(
    dim=16, history=20, targets_per_user=10, batch_size=256, epochs=3, n_tags=50
)
QUICK_RANKER = RankerSettings(n_estimators=40, num_leaves=15, min_child_samples=5)
