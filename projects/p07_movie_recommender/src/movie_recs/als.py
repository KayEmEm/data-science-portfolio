"""Implicit alternating least squares (Hu, Koren and Volinsky 2008) with the ``implicit`` package.

Every rating counts as a view. The user by item matrix is weighted with BM25,
which lowers the weight of very popular movies and very active users.
"""

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from threadpoolctl import threadpool_limits

from movie_recs.config import ALSSettings
from movie_recs.interactions import Stage
from movie_recs.metrics import top_n


def fit(stage: Stage, settings: ALSSettings, seed: int, threads: int = 0) -> AlternatingLeastSquares:
    weighted = (bm25_weight(stage.matrix, K1=100, B=0.8) * settings.alpha).tocsr().astype(np.float32)
    # implicit runs its own threads; a BLAS thread pool inside them slows the fit down.
    with threadpool_limits(1, "blas"):
        model = AlternatingLeastSquares(
            factors=settings.factors,
            regularization=settings.regularization,
            iterations=settings.iterations,
            random_state=seed,
            num_threads=threads,
        )
        model.fit(weighted, show_progress=False)
    return model


def recommend(
    model: AlternatingLeastSquares, stage: Stage, users: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    user_factors = np.asarray(model.user_factors, dtype=np.float32)
    item_factors = np.asarray(model.item_factors, dtype=np.float32)
    return top_n(
        lambda rows: user_factors[rows] @ item_factors.T,
        stage.user_rows(users),
        stage.matrix,
        stage.items,
        n,
    )
