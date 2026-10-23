"""Paths, source schema expectations and run settings for project 4."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CUSTOMERS_PATH = DATA_DIR / "customers.parquet"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

# The original host (blog.minethatdata.com) links the CSV from its 2008 post.
# The copy below is the one scikit-uplift downloads in fetch_hillstrom; the
# checksum pins the exact file this project was run on.
SOURCE_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
SOURCE_SHA256 = "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f"
RAW_NAME = "hillstrom.csv.gz"
EXPECTED_ROWS = 64_000
SEED = 42

# Source schema. The loader checks every column and category before any analysis.
ARM_COL = "segment"
CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]
ARM_LABELS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
OUTCOMES = ["visit", "conversion", "spend"]
ZIP_CODES = ["Urban", "Surburban", "Rural"]  # "Surburban" is the spelling in the source
CHANNELS = ["Phone", "Web", "Multichannel"]
REQUIRED_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    ARM_COL,
    *OUTCOMES,
]
# Pre treatment covariates: everything known before the campaign was sent.
# history_segment is a binned copy of history and is left out.
COVARIATES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
PRE_PERIOD_COL = "history"  # past year spend, the CUPED covariate


@dataclass(frozen=True)
class Economics:
    """Illustrative unit economics for the targeting policy.

    These are assumptions, not values from the dataset. Each email sent costs
    ``cost_per_email`` dollars (delivery, creative and the risk of an
    unsubscribe). Each incremental dollar of spend earns ``margin`` dollars.
    """

    margin: float = 0.30
    cost_per_email: float = 0.10


@dataclass(frozen=True)
class RunSettings:
    """Uplift model settings. ``quick`` shrinks everything for smoke tests."""

    test_size: float = 0.3
    validation_size: float = 0.25  # share of the training split held out for model selection
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 200
    forest_trees: int = 400
    forest_min_leaf: int = 100
    n_bootstrap: int = 500

    @classmethod
    def quick(cls) -> "RunSettings":
        return cls(n_estimators=40, learning_rate=0.1, forest_trees=40, forest_min_leaf=40, n_bootstrap=50)
