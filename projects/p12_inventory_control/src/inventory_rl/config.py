"""Paths, data source, product selection, cost assumptions and training settings for project 12."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

from portfolio_common.config import settings as yaml_settings

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
SEED = 42

# The M5 archive of project 5: the same file and checksum, so both projects read the same data.
SOURCE_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
SOURCE_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
RAW_NAME = "m5.zip"
SOURCE_FILES = {
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
    "train": "sales_train_evaluation.csv",  # d_1 to d_1941
    "test": "sales_test_evaluation.csv",  # d_1942 to d_1969
}
ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EXPECTED_DAYS = 1_969

# Time windows, as 1 based M5 day numbers. The demand model is fitted on the 52 weeks before the
# test year; the real demand of the test year is used only for the replay check in evaluate.py.
TEST_DAYS = 365
FIT_DAYS = 364
TEST_START = EXPECTED_DAYS - TEST_DAYS + 1  # d_1605
FIT_START = TEST_START - FIT_DAYS  # d_1241

# Product selection on the fit window only. Each profile picks the series whose mean daily units
# is closest to a quantile of the mean among the candidates that meet its share of zero days.
# A candidate sold on its first fit day or earlier and has a price on every fit day.
PROFILES = {
    "fast": {"zero_share": (0.0, 0.02), "mean_quantile": 0.99},
    "medium": {"zero_share": (0.0, 0.20), "mean_quantile": 0.75},
    "slow": {"zero_share": (0.50, 0.70), "mean_quantile": 0.50},
}
SAMPLE_STORES = ["CA_1"]  # the stores read when --sample is below 1

DEMAND_PATH = DATA_DIR / "demand.parquet"  # product, d, date, wday, units, sell_price
PRODUCTS_PATH = DATA_DIR / "products.json"  # the selected series, fitted demand and costs

# Cost and supply assumptions. M5 has no cost, stock or supplier data: these are stated choices.
GROSS_MARGIN = 0.40  # unit cost = (1 - margin) x mean sell price
HOLDING_RATE = 0.25  # per year, on the unit cost
ORDER_COST = 10.0  # dollars per order placed, whatever its size
LEAD_TIMES = (2, 3, 4, 5, 6)  # days from order to arrival
LEAD_PROBS = (0.10, 0.30, 0.30, 0.20, 0.10)
CAPACITY_DAYS = 42  # largest inventory position, in days of mean demand
MIN_CAPACITY = 20  # units
HORIZON = 365  # days per episode: one simulated year
GAMMA = 0.99

# Scenario seeds: separate streams for tuning the baselines, choosing RL checkpoints, testing.
TUNE_SEED, VALIDATION_SEED, TEST_SEED, REPLAY_SEED = 30_000, 10_000, 20_000, 40_000


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class RunSettings:
    """Sizes of the runs. ``scaled`` gives the settings of a --sample run."""

    ppo_steps: int = 1_000_000
    sac_steps: int = 200_000
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    shaping_seeds: tuple[int, ...] = (42, 43, 44)
    tune_episodes: int = 200
    validation_episodes: int = 50
    test_episodes: int = 500
    replay_episodes: int = 200
    eval_every: int = 20_000  # training steps between validation checks
    n_envs: int = 8

    def scaled(self, fraction: float) -> "RunSettings":
        if not 0 < fraction <= 1:
            raise ValueError(f"sample must be in (0, 1], got {fraction}")
        if fraction == 1:
            return self

        def count(n: int, low: int) -> int:
            return max(low, math.ceil(n * fraction))

        return replace(
            self,
            ppo_steps=count(self.ppo_steps, 20_000),
            sac_steps=count(self.sac_steps, 5_000),
            seeds=self.seeds[: count(len(self.seeds), 1)],
            shaping_seeds=self.shaping_seeds[: count(len(self.shaping_seeds), 1)],
            tune_episodes=count(self.tune_episodes, 20),
            validation_episodes=count(self.validation_episodes, 10),
            test_episodes=count(self.test_episodes, 50),
            replay_episodes=count(self.replay_episodes, 20),
            eval_every=count(self.eval_every, 2_000),
        )


FULL = RunSettings()
QUICK = RunSettings(
    ppo_steps=4_096,
    sac_steps=1_500,
    seeds=(42,),
    shaping_seeds=(42,),
    tune_episodes=10,
    validation_episodes=4,
    test_episodes=10,
    replay_episodes=4,
    eval_every=1_024,
    n_envs=2,
)


def settings(sample: float = 1.0, quick: bool = False) -> RunSettings:
    return QUICK if quick else FULL.scaled(sample)


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class PPOSettings:
    learning_rate: float = 3e-4
    n_steps: int = 512  # per environment
    batch_size: int = 256
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    net: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class SACSettings:
    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 5_000
    tau: float = 0.005
    net: tuple[int, ...] = (64, 64)


# Reward shaping study: the reward the agent trains on; every policy is scored on the true cost.
SHAPING = {
    "cost": "negative daily cost (main runs)",
    "potential": "plus a potential on the distance of the position to the middle of the tuned (s, S) band",
    "service": "plus an extra penalty equal to the lost margin on each lost unit",
}
POTENTIAL_WEIGHT = 0.5  # potential: -weight x |position - (s + S) / 2| / capacity
SHAPING_PRODUCT = "medium"
ALGORITHMS = ("ppo", "sac")
