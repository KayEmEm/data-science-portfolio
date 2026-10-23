"""Forecast reconciliation over the store and department hierarchy (levels 1 to 9).

Base forecasts for every aggregate are made coherent with the linear
projection of Wickramasuriya, Athanasopoulos and Hyndman (2019):

    reconciled = S (S' W^-1 S)^-1 S' W^-1 base

Methods differ in W: bottom up keeps the level 9 forecasts and sums them; OLS
uses the identity; WLS structural uses the number of bottom series under each
aggregate; MinT shrink uses the in sample one step residual covariance,
shrunk towards its diagonal (Schäfer and Strimmer 2005).
"""

import numpy as np

METHODS = ["bottom_up", "ols", "wls_struct", "mint_shrink"]


def shrunk_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance of residuals (aggregates x days) shrunk to its diagonal; returns it and lambda."""
    e = residuals - residuals.mean(axis=1, keepdims=True)
    t = e.shape[1]
    cov = e @ e.T / t
    sd = np.sqrt(np.diag(cov))
    sd = np.where(sd > 0, sd, 1.0)
    z = e / sd[:, None]
    corr = z @ z.T / t
    # Variance of each correlation estimate (Schäfer and Strimmer, equation for target D)
    w = np.einsum("it,jt->ijt", z, z)
    var_corr = ((w - corr[:, :, None]) ** 2).sum(axis=2) * t / (t - 1) ** 3
    off = ~np.eye(len(cov), dtype=bool)
    lam = float(np.clip(var_corr[off].sum() / (corr[off] ** 2).sum(), 0.0, 1.0))
    shrunk = cov * (1 - lam)
    shrunk[np.diag_indices_from(shrunk)] = np.diag(cov)
    return shrunk, lam


def projection(summing: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The matrix G with reconciled bottom forecasts = G @ base forecasts of all aggregates."""
    w_inv = np.linalg.pinv(weights) if weights.ndim == 2 else np.diag(1.0 / weights)
    left = summing.T @ w_inv
    return np.linalg.solve(left @ summing, left)


def reconcile(
    base: np.ndarray, summing: np.ndarray, method: str, residuals: np.ndarray | None = None
) -> np.ndarray:
    """Coherent forecasts for every aggregate (rows of ``summing``) from base forecasts."""
    n_bottom = summing.shape[1]
    if method == "bottom_up":
        return summing @ base[-n_bottom:]
    if method == "ols":
        weights = np.ones(summing.shape[0])
    elif method == "wls_struct":
        weights = summing.sum(axis=1).astype(float)
    elif method == "mint_shrink":
        if residuals is None:
            raise ValueError("mint_shrink needs in sample residuals")
        weights, _ = shrunk_covariance(residuals)
    else:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")
    return summing @ (projection(summing, weights) @ base)
