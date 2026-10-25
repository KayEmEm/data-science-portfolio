"""Daily demand model: negative binomial with one mean per weekday and one dispersion.

Units on day t follow NB(mean m[w(t)], dispersion r), with variance m + m^2 / r. The weekday
means are the averages of the fit window (store closures left out) and r is the maximum
likelihood estimate given those means; a dispersion at the upper bound is a Poisson model.
Weekdays are numbered as ``wday`` in M5: 1 is Saturday, 7 is Friday.
"""

import numpy as np
from scipy import optimize, stats

MAX_DISPERSION = 1e4


def nb_params(mean: np.ndarray, dispersion: float) -> tuple[float, np.ndarray]:
    """SciPy and NumPy (n, p) for a mean and a dispersion."""
    mean = np.maximum(np.asarray(mean, dtype="float64"), 1e-9)
    return dispersion, dispersion / (dispersion + mean)


def fit(units, wday, closed=None) -> dict:
    units = np.asarray(units, dtype="float64")
    wday = np.asarray(wday, dtype="int64")
    keep = np.ones(len(units), bool) if closed is None else ~np.asarray(closed, bool)
    units, wday = units[keep], wday[keep]
    weekday_mean = np.array([units[wday == w].mean() if (wday == w).any() else units.mean() for w in range(1, 8)])
    weekday_mean = np.maximum(weekday_mean, 1e-3)
    mean = weekday_mean[wday - 1]

    def negative_log_likelihood(log_r: float) -> float:
        n, p = nb_params(mean, float(np.exp(log_r)))
        return -float(stats.nbinom.logpmf(units, n, p).sum())

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=(np.log(1e-2), np.log(MAX_DISPERSION)), method="bounded"
    )
    return {
        "weekday_mean": [float(m) for m in weekday_mean],
        "dispersion": float(np.exp(result.x)),
        "days": int(keep.sum()),
    }


def interval_coverage(model: dict, units, wday, level: float) -> float:
    n, p = nb_params(np.asarray(model["weekday_mean"])[np.asarray(wday) - 1], model["dispersion"])
    low = stats.nbinom.ppf((1 - level) / 2, n, p)
    high = stats.nbinom.ppf((1 + level) / 2, n, p)
    units = np.asarray(units)
    return float(((units >= low) & (units <= high)).mean())


def check(model: dict, frame, test_start: int) -> dict:
    """Compare the model with the fit window (in sample) and with the real test year."""
    out = {}
    for name, part in (("fit", frame[frame["d"] < test_start]), ("test", frame[frame["d"] >= test_start])):
        part = part[~part["closed"].astype(bool)]
        expected = np.asarray(model["weekday_mean"])[part["wday"].to_numpy() - 1]
        out[name] = {
            "days": len(part),
            "mean_units": float(part["units"].mean()),
            "mean_ratio": float(part["units"].mean() / expected.mean()),
            "zero_share": float((part["units"] == 0).mean()),
            "model_zero_share": float(
                stats.nbinom.pmf(0, *nb_params(expected, model["dispersion"])).mean()
            ),
            "coverage_80": interval_coverage(model, part["units"], part["wday"], 0.80),
            "coverage_95": interval_coverage(model, part["units"], part["wday"], 0.95),
        }
    return out


def draw(model: dict, rng: np.random.Generator, n: int, horizon: int, first_wday) -> np.ndarray:
    """(n x horizon) daily units; ``first_wday`` is the weekday of day 0 (one value or one per row)."""
    first = np.broadcast_to(np.asarray(first_wday, dtype="int64"), (n,))
    wday = (first[:, None] - 1 + np.arange(horizon)[None, :]) % 7 + 1
    mean = np.asarray(model["weekday_mean"])[wday - 1]
    r, p = nb_params(mean, model["dispersion"])
    return rng.negative_binomial(r, p).astype("int64")
