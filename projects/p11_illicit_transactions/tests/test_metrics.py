import numpy as np

from illicit_graph.metrics import at_threshold, best_f1_threshold, by_step, step_bootstrap


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.4, 0.8, 0.1, 0.7])
    out = at_threshold(y, s, 0.5)
    assert out["precision"] == 2 / 3
    assert out["recall"] == 2 / 3
    assert np.isclose(out["f1"], 2 / 3)
    assert at_threshold(y, s, 2.0)["f1"] == 0.0


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    t = best_f1_threshold(y, s)
    assert at_threshold(y, s, t)["f1"] == 1.0


def test_by_step_and_bootstrap_use_whole_steps():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(35, 45), 50)
    y = (rng.random(len(steps)) < 0.2).astype(int)
    s = y * 0.6 + rng.random(len(steps)) * 0.5
    rows = by_step(y, s, steps, 0.5)
    assert [r["time_step"] for r in rows] == list(range(35, 45))
    assert sum(r["labelled"] for r in rows) == len(y)
    values = step_bootstrap(y, steps, lambda r: len(r), 30, 1)
    assert set(np.unique(values % 50)) == {0}
