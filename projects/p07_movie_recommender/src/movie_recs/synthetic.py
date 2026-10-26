"""A small synthetic archive in the MovieLens 32M layout, for the tests and ``make smoke``.

Its numbers say nothing about MovieLens. Each user has a taste over the
genres; each movie has one to three genres, a popularity and a date from which
it can be rated. A user picks movies by popularity and taste, rates liked
genres higher, and tags some movies with a word for one of their genres.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from movie_recs.config import ARCHIVE_ROOT, GENRES, SEED, SOURCE_FILES

START = 1_262_304_000  # 2010-01-01
SPAN = 8 * 365 * 86_400
TAG_WORDS = {
    "Comedy": "funny",
    "Horror": "scary",
    "Sci-Fi": "space",
    "Romance": "love",
    "Action": "explosions",
}


def make_tables(n_users: int = 800, n_movies: int = 500, mean_ratings: int = 60, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n_genres = len(GENRES)
    genre_sets = [rng.choice(n_genres, size=rng.integers(1, 4), replace=False) for _ in range(n_movies)]
    genre_matrix = np.zeros((n_movies, n_genres))
    for i, g in enumerate(genre_sets):
        genre_matrix[i, g] = 1.0
    popularity = rng.zipf(1.6, n_movies).clip(max=30).astype(float)
    # Most movies exist from the start; some are released in the last two years, a few in the last
    # two weeks, so that the evaluation windows hold movies without any earlier rating.
    draw = rng.random(n_movies)
    available = np.select(
        [draw < 0.88, draw < 0.95],
        [START, START + rng.integers(SPAN * 3 // 4, SPAN, n_movies)],
        START + rng.integers(SPAN - 14 * 86_400, SPAN - 86_400, n_movies),
    )
    years = np.where(
        available == START, rng.integers(1950, 2010, n_movies), 2016 + rng.integers(0, 2, n_movies)
    )
    taste = rng.normal(0, 1, (n_users, n_genres)) * (rng.random((n_users, n_genres)) < 0.25) * 2.5

    ratings, tags = [], []
    for user in range(n_users):
        n = max(3, int(rng.poisson(mean_ratings) * rng.uniform(0.2, 1.8)))
        first = START + rng.integers(0, SPAN - 86_400)  # some users join during the windows
        times = np.sort(rng.integers(first, START + SPAN, n))
        affinity = genre_matrix @ taste[user] / genre_matrix.sum(1)
        weight = popularity * np.exp(2.0 * affinity)
        chosen = set()
        for t in times:
            open_ = (available <= t) & ~np.isin(np.arange(n_movies), list(chosen))
            if not open_.any():
                break
            p = np.where(open_, weight, 0)
            movie = int(rng.choice(n_movies, p=p / p.sum()))
            chosen.add(movie)
            stars = np.clip(np.round((3.2 + 0.8 * affinity[movie] + rng.normal(0, 0.6)) * 2) / 2, 0.5, 5.0)
            ratings.append((user + 1, movie + 1, stars, int(t)))
            liked = [GENRES[g] for g in genre_sets[movie] if GENRES[g] in TAG_WORDS]
            if liked and rng.random() < 0.15:
                tags.append((user + 1, movie + 1, TAG_WORDS[liked[0]], int(t) + 60))

    titles = [f"Movie {i + 1} ({years[i]})" for i in range(n_movies)]
    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_movies + 1),
            "title": titles,
            "genres": ["|".join(GENRES[g] for g in sorted(gs)) for gs in genre_sets],
        }
    )
    return {
        "ratings": pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"]),
        "movies": movies,
        "tags": pd.DataFrame(tags, columns=["userId", "movieId", "tag", "timestamp"]),
    }


def write_archive(path: Path, **kwargs) -> Path:
    tables = make_tables(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, name in SOURCE_FILES.items():
            buffer = io.StringIO()
            tables[key].to_csv(buffer, index=False)
            archive.writestr(f"{ARCHIVE_ROOT}/{name}", buffer.getvalue())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parse_args(parser, __file__)
    write_archive(args.path)
    print(f"synthetic archive written to {args.path}")


if __name__ == "__main__":
    main()
