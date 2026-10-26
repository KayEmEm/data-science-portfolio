"""Paths, attack budgets and defense settings for project 13.

Project 13 attacks the models of projects 1, 6 and 8. It imports those projects as packages, so
each target is built by the code that built it in its own project: the same features, the same
model settings, the same split rules.
"""

from dataclasses import dataclass
from pathlib import Path

from portfolio_common.config import settings as yaml_settings

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

SEED = 42
PARTS = ("credit", "flows", "images")

# Default share of each data set for ``make sample``. The committed results come from this sample.
SAMPLE = 0.2

# Sources, as each project reads them. ``make data`` takes one folder per part.
CREDIT_SOURCE = RAW_DIR / "credit"  # the UCI zip or spreadsheet of project 1
FLOWS_SOURCE = RAW_DIR / "flows"  # UNSW-NB15 CSV files of project 8 (official partition)
IMAGES_SOURCE = RAW_DIR / "images"  # RDD2022 country zips of project 6
IMAGE_COUNTRIES = ("Czech",)  # India (527 MB) is left out to keep transfer and run time short
# Weights of the project 6 detector: the "full" run trained on a Colab T4 GPU (test mAP50 0.261 in
# project 6). The selected run of project 6 (light augmentation, 0.288) was not kept.
DETECTOR_WEIGHTS = RAW_DIR / "detector" / "best.pt"


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class CreditBudget:
    """What a credit applicant can change before scoring, as a share of the credit limit.

    ``extra_payment`` is the largest extra payment in the latest month: it raises PAY_AMT1 and
    lowers BILL_AMT1 by the same amount. The money is assumed to be borrowed elsewhere for the
    application, so it does not change the true risk ("window dressing"). Self reported fields
    (education, marital status) can take any value the form allows.
    """

    extra_payment: float = 0.10
    payment_steps: int = 11  # grid from 0 to ``extra_payment``
    self_reported: tuple[str, ...] = ("EDUCATION", "MARRIAGE")


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class FlowBudget:
    """What an attacker can change in its own flows without breaking the attack.

    Sizes, packet counts and durations can only grow (padding, extra packets, delays), up to the
    given factors. The initial TTL of sent packets is set by the attacker's host, so ``sttl`` can
    take any value seen in normal training flows. Derived features are recomputed.
    """

    bytes_factor: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    packets_factor: tuple[float, ...] = (1.0, 1.5, 2.0)
    duration_factor: tuple[float, ...] = (1.0, 2.0, 5.0)
    ttl: bool = True


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class ImageBudget:
    """Perturbation limits on images scaled to [0, 1], at the detector input size."""

    epsilons: tuple[float, ...] = (2 / 255, 8 / 255)
    pgd_steps: int = 10
    patch_size: int = 64  # pixels at 512, about 1.6 percent of the image
    patch_steps: int = 60
    patch_batch: int = 8
    patch_lr: float = 0.03


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class SmoothingSettings:
    sigmas: tuple[float, ...] = (0.25, 0.5)
    samples: int = 100
    alpha: float = 0.001  # the certificate holds with probability 1 - alpha
    rows: int = 2000  # test flows certified


IMAGE_SIZE = 512  # the input size of the project 6 detector
CONF = 0.001  # Ultralytics validation threshold, as in project 6
FLOW_TARGET_FPR = 0.01  # alarm threshold of project 8
POISON_TYPE = "Generic"  # attack type the poisoning targets: most of what Isolation Forest finds is Generic
POISON_SHARES = (0.0, 0.01, 0.05, 0.1)
TRIM_SHARE = 0.05  # sanitizing defense: drop this share of the highest scored training flows, refit
ATTACKED_ROWS = 2000  # attack flows that the evasion search tries per model
ART_ROWS = 200  # flows for the unconstrained HopSkipJump reference attack
