"""Paths, source schema expectations and run settings for project 1."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLIENTS_PATH = DATA_DIR / "clients.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SEED = 42
ID_COL = "CLIENT_ID"
TARGET_COL = "TARGET"
EXPOSURE_COL = "LIMIT_BAL"
AGE_COL = "AGE"

# Source column names, as in the UCI spreadsheet header. The loader checks
# them before any feature work, so a change in the source file stops the run.
MONTHS = range(1, 7)
SOURCE_TARGET = "default payment next month"
REQUIRED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    *[f"PAY_{m}" for m in range(2, 7)],
    *[f"BILL_AMT{m}" for m in MONTHS],
    *[f"PAY_AMT{m}" for m in MONTHS],
    SOURCE_TARGET,
]


@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative unit economics for the approval threshold.

    These are assumptions, not values from the dataset. Approving a client
    who defaults loses ``loss_given_default`` times the credit limit.
    Approving a client who repays earns ``margin`` times the credit limit.
    Rejecting a client earns and loses nothing.
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
