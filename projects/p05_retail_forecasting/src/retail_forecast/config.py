"""Paths, source expectations, hierarchy and run settings for project 5."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

# Kaggle hosts the competition files but does not resolve from the build
# environment. Nixtla publishes the same eight files, including the truth of
# the final evaluation window and the official weights, as one zip on GitHub.
# The checksum pins the exact archive this project was run on.
SOURCE_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
SOURCE_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
RAW_NAME = "m5.zip"
SOURCE_FILES = {
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
    "train": "sales_train_evaluation.csv",  # d_1 to d_1941
    "test": "sales_test_evaluation.csv",  # d_1942 to d_1969, the final evaluation window
    "weights": "weights_evaluation.csv",  # official WRMSSE weights for that window
}
ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EXPECTED_SERIES = 30_490
EXPECTED_DAYS = 1_969
SEED = 42

# Processed tables written by ``make data``.
SALES_PATH = DATA_DIR / "sales.parquet"  # one row per series, one column per day
SERIES_PATH = DATA_DIR / "series.parquet"  # the id columns, in the row order of SALES_PATH
CALENDAR_PATH = DATA_DIR / "calendar.parquet"
PRICES_PATH = DATA_DIR / "prices.parquet"  # one row per series, one column per day (forward filled weeks)
OFFICIAL_WEIGHTS_PATH = DATA_DIR / "official_weights.parquet"
DEMAND_DIR = DATA_DIR / "demand"  # long daily demand per hierarchy level, reused by project 12

HORIZON = 28

# The twelve M5 aggregation levels, as the id columns each level groups by.
LEVELS: dict[int, list[str]] = {
    1: [],
    2: ["state_id"],
    3: ["store_id"],
    4: ["cat_id"],
    5: ["dept_id"],
    6: ["state_id", "cat_id"],
    7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"],
    9: ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["state_id", "item_id"],
    12: ["item_id", "store_id"],
}
LEVEL_NAMES = {
    1: "total",
    2: "state",
    3: "store",
    4: "category",
    5: "department",
    6: "state x category",
    7: "state x department",
    8: "store x category",
    9: "store x department",
    10: "item",
    11: "state x item",
    12: "item x store",
}
# Levels 1 to 9 form the store and department hierarchy that the statistical
# models forecast directly; level 9 is its bottom.
STORE_DEPT_LEVEL = 9


@dataclass(frozen=True)
class Backtest:
    """Rolling forecast origins, as the last observed day number (d_<origin>).

    Each backtest window covers the 28 days after its origin. The test window
    is the official M5 evaluation window (d_1942 to d_1969); it is scored once,
    after the model choice on the backtest windows.
    """

    origins: tuple[int, ...] = (1829, 1857, 1885, 1913)
    test_origin: int = 1941
    horizon: int = HORIZON


@dataclass(frozen=True)
class RunSettings:
    """Model settings. ``quick`` shrinks everything for smoke tests."""

    history_days: int = 1095  # days of history the statistical and neural models are fitted on
    gbm_train_days: int = 546  # days of bottom level rows LightGBM is trained on
    gbm_rounds: int = 600
    gbm_learning_rate: float = 0.05
    gbm_leaves: int = 127
    prophet: bool = True
    neural_epochs: int = 40
    interval_levels: tuple[float, ...] = (0.80, 0.95)

    @classmethod
    def quick(cls) -> "RunSettings":
        return cls(history_days=200, gbm_train_days=120, gbm_rounds=40, gbm_leaves=15, neural_epochs=2)
