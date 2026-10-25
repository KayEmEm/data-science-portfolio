import numpy as np
import pandas as pd
from loan_default.features import build_features

from adv_robustness import credit
from adv_robustness.config import CreditBudget


def test_derive_matches_project_1_features(data_dir):
    clients = pd.read_parquet(data_dir / "credit" / "clients.parquet")
    ours = credit.derive(clients)
    theirs = build_features(data_dir / "credit" / "clients.parquet")
    pd.testing.assert_frame_equal(ours.reset_index(drop=True), theirs.reset_index(drop=True))


def test_candidates_cover_the_budget_and_keep_totals(data_dir):
    clients = pd.read_parquet(data_dir / "credit" / "clients.parquet").head(5)
    budget = CreditBudget()
    pool = credit.candidates(clients, "both", budget)
    assert len(pool) == 5 * budget.payment_steps * len(credit.EDUCATION_VALUES) * len(credit.MARRIAGE_VALUES)
    assert pool[credit.ID_COL].is_unique
    original = clients.set_index(credit.ID_COL).loc[pool["client"]]
    # The extra payment moves money from the bill to the payment; their sum stays the same.
    np.testing.assert_allclose(
        pool["PAY_AMT1"].to_numpy() + pool["BILL_AMT1"].to_numpy(),
        original["PAY_AMT1"].to_numpy() + original["BILL_AMT1"].to_numpy(),
    )
    assert (pool["cost"] <= budget.extra_payment * original[credit.EXPOSURE_COL].to_numpy() + 1).all()
    payment_only = credit.candidates(clients, "payment", budget)
    assert set(payment_only["EDUCATION"]) == set(clients["EDUCATION"])


def test_attack_finds_cheapest_flip():
    class Threshold:
        """Approves when the latest payment reaches 5000."""

        def predict(self, features):
            return np.where(features["PAY_AMT1"] >= 5000, 0.1, 0.9)

    clients = pd.DataFrame(
        {
            credit.ID_COL: [1, 2],
            credit.TARGET_COL: [1, 0],
            "LIMIT_BAL": [100_000, 10_000],
            "SEX": [1, 2],
            "EDUCATION": [2, 2],
            "MARRIAGE": [1, 1],
            "AGE": [30, 40],
            **{c: [0, 0] for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
            **{f"BILL_AMT{m}": [20_000, 20_000] for m in range(1, 7)},
            **{f"PAY_AMT{m}": [0, 0] for m in range(1, 7)},
        }
    )
    found = credit.attack(Threshold(), 0.5, clients, "payment", CreditBudget())
    # Client 1 needs 5% of a 100,000 limit; client 2 cannot reach 5000 with 10% of 10,000.
    assert found["flipped"].tolist() == [True, False]
    assert found.loc[0, "cost"] == 5000


def test_run_writes_every_variant(data_dir, tmp_path):
    summary = credit.run(data_dir, tmp_path, quick=True)
    assert set(summary["variants"]) == set(credit.VARIANTS)
    for result in summary["variants"].values():
        assert set(result["attacks"]) == set(credit.ATTACK_SETS)
        assert 0 <= result["attacks"]["both"]["flip_rate"] <= 1
        # Changing more cannot turn fewer rejections.
        assert result["attacks"]["both"]["flipped"] >= result["attacks"]["payment"]["flipped"]
    assert (tmp_path / "credit" / "results.json").exists()
