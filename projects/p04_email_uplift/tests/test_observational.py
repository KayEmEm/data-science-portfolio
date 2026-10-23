import numpy as np

from email_uplift.data import normalize
from email_uplift.observational import observational_sample, selection_probability
from email_uplift.synthetic import make_customers


def test_selection_creates_confounding():
    frame = normalize(make_customers(20000), expected_rows=None)
    sample = observational_sample(frame)
    mailed, not_mailed = sample[sample["email"]], sample[~sample["email"]]
    assert mailed["log_history"].mean() > not_mailed["log_history"].mean() + 0.2
    assert mailed["recency"].mean() < not_mailed["recency"].mean()
    p = selection_probability(frame)
    assert 0 < p.min() < 0.2 and 0.8 < p.max() < 1


def test_selection_is_reproducible():
    frame = normalize(make_customers(3000), expected_rows=None)
    first, again = observational_sample(frame, seed=7), observational_sample(frame, seed=7)
    assert first.equals(again)
    assert not np.array_equal(first["visit"], observational_sample(frame, seed=8)["visit"])
