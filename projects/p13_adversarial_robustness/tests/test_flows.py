import numpy as np
import pandas as pd
from flow_anomaly.data import load

from adv_robustness import flows
from adv_robustness.config import FlowBudget


def test_perturb_only_grows_and_scales_derived_features(data_dir):
    sample = load(data_dir / "flows").head(200)
    out = flows.perturb(sample, 2.0, 2.0, 5.0, ttl=31)
    for column in ("sbytes", "spkts", "dur"):
        assert (out[column] >= sample[column]).all()
    assert (out["sttl"] == 31).all()
    positive = sample["dur"] > 0
    np.testing.assert_allclose(out.loc[positive, "sload"], sample.loc[positive, "sload"] * 2 / 5)
    assert out["dbytes"].equals(sample["dbytes"])


def test_unchanged_flow_and_grid_size():
    budget = FlowBudget()
    sizes = len(budget.bytes_factor) * len(budget.packets_factor) * len(budget.duration_factor)
    assert len(flows.grid(budget, [31, 62], "size_timing")) == sizes
    assert len(flows.grid(budget, [31, 62], "ttl")) == 3
    assert len(flows.grid(budget, [31, 62], "all")) == sizes * 3
    one = pd.DataFrame(
        {
            "sbytes": [100.0],
            "spkts": [2.0],
            "dur": [1.0],
            "smean": [50.0],
            "sload": [800.0],
            "sinpkt": [1000.0],
            "sjit": [0.0],
            "sttl": [254],
        }
    )
    assert flows.perturb(one, 1.0, 1.0, 1.0).equals(one.astype(flows.perturb(one, 1.0, 1.0, 1.0).dtypes))
    padded = flows.perturb(one, 40.0, 1.0, 1.0)
    assert not flows.feasible(padded)[0]  # 2,000 bytes per packet exceed the MTU


def test_evasion_never_raises_the_score(data_dir):
    table = load(data_dir / "flows")

    class SumScore:
        threshold = 0.0

        def score(self, frame):
            return frame["sbytes"].to_numpy() / (1 + frame["dur"].to_numpy())

    attacks = table[table["label"] == 1].head(50)
    found = flows.evade(SumScore(), attacks, FlowBudget(), [31], "all")
    assert (found["score"].to_numpy() <= SumScore().score(attacks) + 1e-9).all()


def test_run_quick(data_dir, tmp_path):
    summary = flows.run(data_dir, tmp_path, quick=True)
    assert set(summary["models"]) == {"lightgbm", "lightgbm_no_ttl", "lightgbm_adversarial", "iforest"}
    # Without the TTL features, changing the TTL cannot evade.
    assert summary["models"]["lightgbm_no_ttl"]["evasion"]["groups"]["ttl"]["evasion_rate"] == 0.0
    assert set(summary["poisoning"]) == {"0", "0.01", "0.05", "0.1"}
    for result in summary["smoothing"].values():
        certified = list(result["certified_accuracy"].values())
        assert certified == sorted(certified, reverse=True)
