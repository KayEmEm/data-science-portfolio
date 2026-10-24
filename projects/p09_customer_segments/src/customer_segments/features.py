"""Customer features up to the cutoff, the outcome after it, and the scaled matrices for clustering.

The cutoff is the start of the day ``OUTCOME_DAYS`` before the last invoice. A customer enters the
table when they bought at least one product before the cutoff and their net product spend before it
is positive. Features, all from product lines before the cutoff:

* ``recency_days``: days from the last purchase to the cutoff;
* ``frequency``: number of purchase invoices;
* ``monetary``: net spend (purchases less returns), in pounds sterling;
* ``mean_basket``: gross purchase value per invoice;
* ``tenure_days``: days from the first purchase to the cutoff;
* ``distinct_products``: number of different stock codes bought;
* ``return_share``: returned value as a share of gross purchase value, capped at 1;
* ``share_<category>``: share of gross purchase value per product category.

Outcome, from the window after the cutoff: ``future_spend`` (net product spend, at least 0) and
``future_purchase`` (1 when the customer bought anything).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from customer_segments.categories import CATEGORIES
from customer_segments.config import BEHAVIOR, CLUSTER_BEHAVIOR, LOG_FEATURES, MIX_WEIGHT, OUTCOME_DAYS, RFM

SHARES = [f"share_{c}" for c in CATEGORIES]


def cutoff_date(lines: pd.DataFrame, outcome_days: int = OUTCOME_DAYS) -> pd.Timestamp:
    return (lines["invoice_date"].max() - pd.Timedelta(days=outcome_days)).normalize()


def customer_table(lines: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    products = lines.loc[lines["is_product"]]
    before = products.loc[products["invoice_date"] < cutoff]
    buys = before.loc[~before["is_return"]]
    returns = before.loc[before["is_return"]]

    grouped = buys.groupby("customer_id")
    table = pd.DataFrame(
        {
            "first_purchase": grouped["invoice_date"].min(),
            "last_purchase": grouped["invoice_date"].max(),
            "frequency": grouped["invoice"].nunique(),
            "gross": grouped["value"].sum(),
            "distinct_products": grouped["stock_code"].nunique(),
            "country": grouped["country"].agg(lambda s: s.mode().iat[0]),
        }
    )
    returned = (-returns.groupby("customer_id")["value"].sum()).reindex(table.index, fill_value=0.0)
    table["monetary"] = table["gross"] - returned
    table["recency_days"] = (cutoff - table["last_purchase"]).dt.total_seconds() / 86400
    table["tenure_days"] = (cutoff - table["first_purchase"]).dt.total_seconds() / 86400
    table["mean_basket"] = table["gross"] / table["frequency"]
    table["return_share"] = (returned / table["gross"]).clip(0, 1)

    mix = buys.pivot_table(index="customer_id", columns="category", values="value", aggfunc="sum")
    mix = mix.reindex(index=table.index, columns=CATEGORIES).fillna(0.0)
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    mix.columns = SHARES
    table = table.join(mix)

    after = products.loc[products["invoice_date"] >= cutoff]
    future = after.groupby("customer_id")["value"].sum().reindex(table.index, fill_value=0.0)
    bought = after.loc[~after["is_return"]].groupby("customer_id").size()
    table["future_spend"] = future.clip(lower=0)
    table["future_purchase"] = bought.reindex(table.index, fill_value=0).gt(0).astype("int8")

    table = table.loc[table["monetary"] > 0]
    columns = BEHAVIOR + SHARES + ["country", "first_purchase", "last_purchase", "future_spend"]
    return table[columns + ["future_purchase"]].sort_index()


def log_transform(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in LOG_FEATURES:
        if column in out:
            out[column] = np.log1p(out[column])
    return out


def scaler(name: str, n_rows: int):
    if name in ("raw", "standard"):
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, n_rows), random_state=0
        )
    raise ValueError(f"unknown scaling {name}")


def scaled_block(frame: pd.DataFrame, scaling: str) -> np.ndarray:
    """``raw`` scales the features as they are; the others log transform the skewed ones first."""
    values = frame if scaling == "raw" else log_transform(frame)
    return scaler(scaling, len(frame)).fit_transform(values.to_numpy(dtype=float))


def matrix(table: pd.DataFrame, feature_set: str, scaling: str) -> np.ndarray:
    """The clustering input: the behavior block, and for the mix sets the weighted category block.

    ``rfm_mix`` uses the behavior features without ``return_share``; ``rfm_mix_returns`` uses all.
    """
    if feature_set == "rfm":
        return scaled_block(table[RFM], scaling)
    if feature_set not in ("rfm_mix", "rfm_mix_returns"):
        raise ValueError(f"unknown feature set {feature_set}")
    columns = BEHAVIOR if feature_set == "rfm_mix_returns" else CLUSTER_BEHAVIOR
    behavior = scaled_block(table[columns], scaling)
    shares = table[SHARES].to_numpy(dtype=float)
    shares = shares[:, shares.std(axis=0) > 0]
    mix = StandardScaler().fit_transform(shares)
    # The weight sets the total variance of the category block to MIX_WEIGHT times that of the behavior block.
    weight = np.sqrt(MIX_WEIGHT * behavior.var(axis=0).sum() / max(mix.var(axis=0).sum(), 1e-12))
    return np.hstack([behavior, weight * mix])
