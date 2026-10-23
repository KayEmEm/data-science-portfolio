"""Paths, table schema expectations and run settings for project 1."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_COMPETITION = "home-credit-default-risk"
SEED = 42
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# Columns each source table must contain. The loader checks them before any
# feature work, so a schema change in the source data stops the run early.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "application_train": [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ],
    "bureau": [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
    ],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    "previous_application": [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "DAYS_DECISION",
    ],
    "POS_CASH_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"],
    "installments_payments": [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ],
    "credit_card_balance": [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "SK_DPD",
    ],
}


@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative unit economics for the approval threshold.

    These are assumptions, not values from the dataset. Approving a borrower
    who defaults loses ``loss_given_default`` times the credit amount.
    Approving a borrower who repays earns ``margin`` times the credit amount.
    Rejecting a borrower earns and loses nothing.
    """

    loss_given_default: float = 0.45
    margin: float = 0.10


@dataclass(frozen=True)
class RunSettings:
    """Model selection settings. ``quick`` shrinks everything for smoke tests."""

    n_folds: int = 5
    test_size: float = 0.2
    n_estimators: int = 2000
    learning_rate: float = 0.03
    early_stopping_rounds: int = 100
    shap_sample: int = 2000

    @classmethod
    def quick(cls) -> "RunSettings":
        return cls(n_folds=3, n_estimators=60, learning_rate=0.1, early_stopping_rounds=10, shap_sample=200)
