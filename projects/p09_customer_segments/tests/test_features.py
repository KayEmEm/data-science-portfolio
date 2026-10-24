import numpy as np
import pandas as pd

from customer_segments.config import BEHAVIOR, OUTCOME_DAYS
from customer_segments.features import SHARES, customer_table, cutoff_date, matrix


def test_cutoff_and_table(lines):
    cutoff = cutoff_date(lines)
    assert cutoff == (lines["invoice_date"].max() - pd.Timedelta(days=OUTCOME_DAYS)).normalize()
    table = customer_table(lines, cutoff)
    assert (table["monetary"] > 0).all()
    assert (table["recency_days"] >= 0).all() and (table["tenure_days"] >= table["recency_days"]).all()
    assert np.allclose(table[SHARES].sum(axis=1), 1)
    assert table["return_share"].between(0, 1).all()
    assert set(table["future_purchase"].unique()) <= {0, 1}
    assert (table["future_spend"] >= 0).all()


def test_features_use_only_lines_before_the_cutoff(lines):
    cutoff = cutoff_date(lines)
    table = customer_table(lines, cutoff)
    # Moving every later line further into the future changes no feature.
    shifted = lines.copy()
    later = shifted["invoice_date"] >= cutoff
    shifted.loc[later, "invoice_date"] += pd.Timedelta(days=30)
    again = customer_table(shifted, cutoff)
    pd.testing.assert_frame_equal(table[BEHAVIOR + SHARES], again[BEHAVIOR + SHARES])


def test_matrices(lines):
    table = customer_table(lines, cutoff_date(lines))
    rfm = matrix(table, "rfm", "standard")
    assert rfm.shape == (len(table), 3)
    assert np.allclose(rfm.mean(axis=0), 0, atol=1e-9)
    mix = matrix(table, "rfm_mix", "robust")
    behavior = mix[:, : len(BEHAVIOR)].var(axis=0).sum()
    assert np.isclose(mix[:, len(BEHAVIOR) :].var(axis=0).sum(), 0.5 * behavior)
    for scaling in ("raw", "quantile"):
        assert np.isfinite(matrix(table, "rfm_mix", scaling)).all()
